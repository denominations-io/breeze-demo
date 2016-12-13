import org.apache.spark.sql._

case class HitRate(truePositives: Long, falsePositives: Long, hitRate: Double)

trait EvaluationMetrics {

  def computeRMSE: Dataset[Prediction] => Double = { predictions =>
    val numObservations = predictions.count()
    import predictions.sparkSession.implicits._
    math.sqrt(
      predictions
        .map { pred => math.pow(pred.predicted - pred.observed, 2) }
        .collect()
        .sum
        / numObservations)
  }

  def computeHitRate: Dataset[Prediction] => HitRate = {
    predictions =>
      import predictions.sparkSession.implicits._
      val numPredictions = predictions.count()
      val truePositives = predictions
        .map { prediction => prediction.predicted match {
          case prob if prob > 0.5 => Prediction(prediction.observed, 1)
          case _                  => Prediction(prediction.observed, 0)
        }}
        .filter { prediction => prediction.observed != prediction.predicted }
        .count()
      HitRate(truePositives, numPredictions - truePositives, truePositives.toDouble / numPredictions.toDouble )
  }

}
