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

class LogisticRegressionWithAdaGrad(spark: SparkSession, data: Dataset[LabeledPoint]) extends Regression
  with ModelEvaluation
  with FeatureSpace
  with AdaGradOptimizer
  with SerializableLogging {

  val modelProperties = Model("Logistic Regression with AdaGrad")

  val numFeatures = data.take(1).map { _.features.size }.head
  val numObservations = data.count()
  logger.info(s"Found $numObservations observations of $numFeatures features.")

  import spark.implicits._

  val labels = new DenseVector[Double](data.map(_.label).collect())
  val featureSpace = featureMatrixFromLabeledPoint(numFeatures, data.collect())

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

  val optimizedParameters = optimizeAGD(regression, DenseVector.zeros[Double](numFeatures))
  logger.info(s"Parameter estimates after optimization: ${optimizedParameters.toString}")

  // TODO: compute the p values of the coefficients
  override def estimate = {
    val fit = Some(ModelFit("AIC", 1200))
    val coefficients =
      Some(optimizedParameters
        .data.map(p => Estimate("a", p, 0.0)).sortBy(-_.probability))
    modelProperties.copy(modelFit = fit, parameterEstimates = coefficients)
  }

  override def evaluation = { holdOut: Dataset[LabeledPoint] =>

    val predictions = holdOut.collect().map { lp =>
      val features = new breeze.linalg.DenseVector[Double](lp.features.toArray)
      val evaluation = sigmoid(sum(features :* optimizedParameters))
      val prediction: Double = evaluation / (1 + evaluation) // TODO: intercept is missing
      Prediction(observed = lp.label, predicted = prediction)
    }

    val predictionDS = holdOut.sparkSession.createDataset(predictions)
    evaluateBinaryModel(estimate, predictionDS)
  }
}

