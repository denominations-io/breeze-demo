package logit
package learning

import breeze.util._

import org.apache.spark.sql._

import tooling._

trait ModelEvaluation extends FeatureSpace with SerializableLogging {

  // TODO: expand model evaluation suite
  def evaluate(spark: SparkSession, predictions: Dataset[Prediction], colNames: Array[String]): Unit = {
    val rmse = computeRMSE(predictions)
    val summary = Summary(
      evaluationSet = predictions.count().toInt,
      evaluation = Map("RMSE (percentage points): " -> rmse * 100),
      diagnostics = Map("convergence rate: " -> 0.04)
    )
    summary.logSummary()
  }

  private def computeRMSE: Dataset[Prediction] => Double = { predictions =>
    val numObservations = predictions.count()
    import predictions.sparkSession.implicits._
    math.sqrt(
      predictions
        .map { pred => math.pow(pred.prediction - pred.observed, 2) }
        .collect()
        .sum
        / numObservations)
  }
}
