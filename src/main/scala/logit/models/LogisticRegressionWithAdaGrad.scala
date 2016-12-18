import breeze.linalg._
import breeze.numerics._
import breeze.optimize._
import breeze.util._

import org.apache.spark.sql._
import org.apache.spark.ml.feature.LabeledPoint

import logit.optimizers._

/**
  * Estimate the logistic function through adaptive gradient descent (Duchi et al., 2010).
  *    the objective function:      p(y|x,b) = 1/(1+exp(-b'x))
  *    the adaptive learning rate:  b_t_i = b_t_i-1 - (eta / sqrt(sum(pow(gradient_ii, 2)) * gradient_ti)
  *    regression with mle:         arg max_b sum(log p(y|x,b)-aR(b)) over m iterations,
  * with R(b) the L2 regularization term R(b) = sum (pow(b,2)) for all n observations.
  */

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
      val prediction: Double = evaluation / (1 + evaluation) // TODO: intercept is missing
      Prediction(observed = lp.label, predicted = prediction)
    }
  }

  val parameterEstimates = DenseVector(estimate.coefficients.map { _.estimate })
  def evaluate = evaluateBinaryPredictions(estimate, predict(parameterEstimates, holdout))
}

