import breeze.util._

import org.apache.spark.sql._

case class Prediction(observed: Double, predicted: Double)

trait ModelEvaluation extends FeatureSpace with EvaluationMetrics with SerializableLogging {

  // TODO: expand model evaluation suite
  def evaluateLinearModel(model: Model, predictions: Dataset[Prediction]): Evaluation = {
    val rmse = computeRMSE(predictions)
    Evaluation(
      modelName = model.name,
      evaluationSet = predictions.count().toInt,
      evaluation = Map("RMSE" -> rmse),
      diagnostics = Map("convergence rate: " -> 0.04)
    )
  }

  def evaluateBinaryModel(model: Model, predictions: Dataset[Prediction]): Evaluation = {
    val hitRate = computeHitRate(predictions)
   Evaluation(
      modelName = model.name,
      evaluationSet = predictions.count().toInt,
      evaluation = Map(
        "True positives: " -> hitRate.truePositives,
        "False positives: " -> hitRate.falsePositives,
        "Hit rate (%): " -> hitRate.hitRate * 100
      ),
      diagnostics = Map("convergence rate: " -> 0.04)
    )
  }
}

case class Evaluation(
                       modelName: String = "",
                       evaluationSet: Int = 0,
                       variables: Int = 0,
                       evaluation: Map[String, Double] = Map.empty[String, Double],
                       diagnostics: Map[String, Double] = Map.empty[String, Double]
                     ) extends SerializableLogging {
  def logSummary(): Unit = {
    logger.info(
      s"""
         |
         |    $modelName
         | evaluation set size:   ${evaluationSet.toString}
         | number of variables:   ${variables.toString}
         |
         | -- predictive performance --
         | ${evaluation.mkString("\n ")}
         |
         | -- diagnostic information --
         | ${diagnostics.mkString("\n ")}
       """.stripMargin
    )
  }
}


