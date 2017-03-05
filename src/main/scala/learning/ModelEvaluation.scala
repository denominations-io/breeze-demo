import java.io._

import org.apache.spark.sql._
import org.apache.spark.mllib.evaluation._

import breeze.util._

case class Prediction(observed: Double, predicted: Double)

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
     val metrics = new BinaryClassificationMetrics(predictions.rdd.map { pred => (pred.observed, pred.predicted) } )

     Evaluation(
        description = model.name,
        evaluationSet = predictions.count().toInt,
        numVariables = model.coefficients.length,
        evaluationMetrics = Map(
          "True positives: " -> hitRate.truePositives,
          "False positives: " -> hitRate.falsePositives,
          "Hit rate (%): " -> hitRate.hitRate * 100,
          "Area under ROC curve: " -> metrics.areaUnderROC()
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
  def generateSummary(fileName: String): Unit = {

    val formattedCoefficients = coefficients.map { c =>
        val variableName = c.variableName.size match {
          case size if size <= 20   => c.variableName + " " * (20 - size)
          case size if size > 20    => c.variableName.substring(0, 20)
        }
        s"$variableName | ${c.estimate.toString.substring(0, 15 )}"
      }

    val eval =
      s"""
       | Model evaluation, summary of the run.
       | Model: $description
       |
       | evaluation set size:   ${evaluationSet.toString}
       | number of variables:   ${numVariables.toString}
       |
       | ${evaluationMetrics.mkString("\n ")}
       |
       |   variable name          coefficient
       | ${formattedCoefficients.mkString("\n ")}
     """.stripMargin

    val file = new File(fileName)
    if(fileName.contains("/")) file.getParentFile.mkdirs()
    new PrintWriter(file) {
      write(eval)
      close
    }
  }
}


