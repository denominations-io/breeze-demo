import breeze.util._

import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.sql._
import org.apache.spark.mllib.classification._
import org.apache.spark.mllib.optimization._

class MLlibLogisticRegressionWithSGD(colNames: Array[String], training: Dataset[LabeledPoint], holdout: Dataset[LabeledPoint]) extends Regression
  with ModelEvaluation
  with SerializableLogging {

  override def description = "Logistic regression with SGD"

  import training.sparkSession.implicits._

  val rdd = training.map { lp => (lp.label, org.apache.spark.mllib.linalg.Vectors.fromML(lp.features)) }.rdd

  val logisticGradient = new LogisticGradient
  val l2Updater = new SquaredL2Updater

  val numVariables = rdd.take(1).map(_._2.size).head
  val initialWeights = org.apache.spark.mllib.linalg.Vectors.zeros(numVariables)

  val (parameterEstimates, loss) =
    GradientDescent.runMiniBatchSGD(rdd, logisticGradient, l2Updater, 8.5, 200, 4, 1, initialWeights, 1E-8)

  println(s"no of estimates: ${parameterEstimates.size}")

  logger.info(s"found ${parameterEstimates.size} weights with intercept: ${parameterEstimates.toString}")
  val regression = new LogisticRegressionModel(parameterEstimates, 0.0, numVariables, 2).setThreshold(1E-10)

  override def estimate: ModelSummary = {
    val modelFit = ModelFit("num features", regression.numFeatures)
    val coefficients = colNames.zipWithIndex.map { variable => Coefficient(variable._1, parameterEstimates(variable._2))}
    ModelSummary(description, modelFit, coefficients)
  }

  val predict: (LogisticRegressionModel, Dataset[LabeledPoint]) => Dataset[Prediction] = {
    (model, holdOut) =>
      holdOut.map { lp =>
        val prediction = model.predict(org.apache.spark.mllib.linalg.Vectors.fromML(lp.features))
        Prediction(observed = lp.label, predicted = prediction)
      }
  }

  def evaluate = evaluateBinaryPredictions(estimate, predict(regression, holdout))
}
