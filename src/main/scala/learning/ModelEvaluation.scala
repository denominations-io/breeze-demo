import breeze.util._

import org.apache.spark.sql._

case class Prediction(observed: Double, predicted: Double)

// TODO: expand model evaluation suite
trait ModelEvaluation extends FeatureSpace with EvaluationMetrics with SerializableLogging {

  def evaluateLinearPredictions(model: ModelSummary, predictions: Dataset[Prediction]): Evaluation = {
    val rmse = computeRMSE(predictions)
    Evaluation(
      description = model.name,
      evaluationSet = predictions.count().toInt,
      numVariables = model.coefficients.length,
      evaluationMetrics = Map("RMSE" -> rmse, "R2" -> model.modelFit.value),
      coefficients = model.coefficients
    )
  }

  def evaluateBinaryPredictions(model: ModelSummary, predictions: Dataset[Prediction]): Evaluation = {
    val hitRate = computeHitRate(predictions)

   Evaluation(
      description = model.name,
      evaluationSet = predictions.count().toInt,
      numVariables = model.coefficients.length,
      evaluationMetrics = Map(
        "True positives: " -> hitRate.truePositives,
        "False positives: " -> hitRate.falsePositives,
        "Hit rate (%): " -> hitRate.hitRate * 100
      ),
     coefficients = model.coefficients
    )
  }
}

case class Evaluation(
                       description: String = "",
                       evaluationSet: Int = 0,
                       numVariables: Int = 0,
                       evaluationMetrics: Map[String, Double] = Map.empty[String, Double],
                       coefficients: Array[Coefficient]
                     ) extends SerializableLogging {
  def logSummary(): Unit = {
    logger.info(
      s"""
         |
         | -$description-
         | evaluation set size:   ${evaluationSet.toString}
         | number of variables:   ${numVariables.toString}
         |
         | -- evaluation metrics --
         | ${evaluationMetrics.mkString("\n ")}
         |
         | --   variable name   ||   coefficient   ||   p-value --
         | ${coefficients.map { c => s"${c.variableName}  |  ${c.estimate}  |  ${c.probability}"}.mkString("\n ")}
       """.stripMargin
    )
  }
}


