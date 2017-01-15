import breeze.linalg._
import breeze.numerics._
import breeze.optimize._
import breeze.util._

import org.apache.spark.sql._
import org.apache.spark.ml.feature.LabeledPoint
import logit.optimizers._

/** Estimate the logistic function through adaptive gradient descent with L2 regularization. */
class LogisticRegressionWithAdaGrad(spark: SparkSession, colNames: Array[String], training: Dataset[LabeledPoint], holdout: Dataset[LabeledPoint]) extends Regression
  with ModelEvaluation
  with FeatureSpace
  with AdaGradOptimizer
  with SerializableLogging {

  override def description = "Logistic Regression with AdaGrad"

  val numFeatures = training.take(1).map { _.features.size }.head
  val numObservations = training.count()
  logger.info(s"Found $numObservations observations of $numFeatures features.")

  import spark.implicits._

  val labels = new DenseVector[Double](training.map(_.label).collect())
  val featureSpace = featureMatrixFromLabeledPoint(numFeatures, training.collect())

  def objectiveFunction(parameters: DenseVector[Double]): Double = {
    val quotient = featureSpace * parameters
    val expQ: DenseVector[Double] = exp(quotient)
    - sum((labels :* quotient) - log1p(expQ))
  }

  def gradient(parameters: DenseVector[Double]): DenseVector[Double] = {
    val quotient = featureSpace * parameters
    val probabilities = sigmoid(quotient)
    featureSpace.t * (probabilities - labels)
  }

  val regression = new DiffFunction[DenseVector[Double]] {
    def calculate(parameters: DenseVector[Double]) = (objectiveFunction(parameters), gradient(parameters))
  }

  // TODO: compute the p values of the coefficients
  def estimate = {
    val optimizedParameters = optimizeAGD(regression, DenseVector.zeros[Double](numFeatures))
    logger.info(s"Parameter estimates after optimization: ${optimizedParameters.toString}")

    val fit = ModelFit("AIC", 0.0)
    val estimates = optimizedParameters.data.zip(colNames).map(p => Coefficient(p._2, p._1, 0.0)).sortBy(-_.probability)
    ModelSummary(description, fit, estimates)
  }

  val predict = { (coefficients: DenseVector[Double], holdOut: Dataset[LabeledPoint]) =>
    holdOut.map { lp =>
      val features = new breeze.linalg.DenseVector[Double](lp.features.toArray)
      val evaluation = sigmoid(sum(features :* coefficients))
      val prediction = evaluation / (1 + evaluation) // TODO: intercept is missing
      Prediction(observed = lp.label, predicted = prediction)
    }
  }

  val parameterEstimates = DenseVector(estimate.coefficients.map { _.estimate })
  def evaluate = evaluateBinaryPredictions(estimate, predict(parameterEstimates, holdout))
}

