package logit
package learning

import breeze.linalg._
import breeze.util._

import org.apache.spark.sql._
import org.apache.spark.ml.feature._

trait Model {
  def estimate: DenseVector[Double]
  def predict: Dataset[LabeledPoint] => Dataset[Prediction]
  def summarize: Summary
}
case class Prediction(prediction: Double, observed: Double)

case class Summary(
                    val evaluationSet: Int = 0,
                    val variables: Int = 0,
                    val evaluation: Map[String, Double] = Map.empty[String, Double],
                    val diagnostics: Map[String, Double] = Map.empty[String, Double]
                  ) extends SerializableLogging {
  def logSummary(): Unit = {
    logger.info(this.toString)
  }
}

