import breeze.util._

import org.apache.spark.sql._
import org.apache.spark.mllib.classification._
import org.apache.spark.mllib.optimization._

class MLLibLogisticRegressionWithSGD(data: Dataset[org.apache.spark.ml.feature.LabeledPoint]) extends Regression
  with ModelEvaluation
  with SerializableLogging {

  val modelProperties = Model("Logistic regression with SGD")

  import data.sparkSession.implicits._

  val rdd = data.map { lp => (lp.label, org.apache.spark.mllib.linalg.Vectors.fromML(lp.features)) }.rdd

  val logisticGradient = new LogisticGradient(2)
  val l2Updater = new SquaredL2Updater

  val numVariables = rdd.take(1).map(_._2.size).head
  val initialWeights = org.apache.spark.mllib.linalg.Vectors.zeros(numVariables)

  val (parameterEstimates, loss) =
    GradientDescent.runMiniBatchSGD(rdd, logisticGradient, l2Updater, 0.000001, 200, 4, 1, initialWeights, 1E-10)

  logger.info(s"found ${parameterEstimates.size} weights with intercept: ${parameterEstimates.toString}")
  val model = new LogisticRegressionModel(parameterEstimates, 0.0, numVariables, 2).setThreshold(1E-10)

  override def estimate: Model = { modelProperties }
  override def evaluation = { holdOut: Dataset[org.apache.spark.ml.feature.LabeledPoint] =>
    val predictions = holdOut.map { lp =>
      val prediction = model.predict(org.apache.spark.mllib.linalg.Vectors.fromML(lp.features))
      Prediction(observed = lp.label, predicted = prediction)
    }
    evaluateBinaryModel(estimate, predictions)
  }
}
