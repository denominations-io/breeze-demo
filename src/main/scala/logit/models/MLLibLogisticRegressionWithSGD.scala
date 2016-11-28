package logit
package models

import breeze.linalg._
import breeze.util._

import org.apache.spark.sql._
import org.apache.spark.mllib.classification._
import org.apache.spark.mllib.optimization._

import learning._

class MLLibLogisticRegressionWithSGD(data: Dataset[org.apache.spark.ml.feature.LabeledPoint]) extends Model with SerializableLogging {

  import data.sparkSession.implicits._

  val rdd = data.map { lp => (lp.label, org.apache.spark.mllib.linalg.Vectors.fromML(lp.features)) }.rdd

  val logisticGradient = new LogisticGradient(2)
  val l2Updater = new SquaredL2Updater

  val numVariables = rdd.take(1).map(_._2.size).head
  val initialWeights = org.apache.spark.mllib.linalg.Vectors.zeros(numVariables)

  val (parameterEstimates, loss) =
    GradientDescent.runMiniBatchSGD(rdd, logisticGradient, l2Updater, 25, 100, 2, 1, initialWeights, 1E-4)

  logger.info(s"found ${parameterEstimates.size} weights with intercept: ${parameterEstimates.toString}")
  val model = new LogisticRegressionModel(parameterEstimates, 0.0, numVariables, 2).setThreshold(1E-4)

  override def estimate: DenseVector[Double] = { DenseVector(parameterEstimates.toArray) }
  override def predict = { holdOut: Dataset[org.apache.spark.ml.feature.LabeledPoint] =>
      holdOut.map { lp =>
        val prediction = model.predict(org.apache.spark.mllib.linalg.Vectors.fromML(lp.features))
        Prediction(prediction, lp.label)
      }
  }
  override def summarize: Summary = { Summary() }

}
