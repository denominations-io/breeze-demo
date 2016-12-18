import org.apache.spark.sql._
import org.scalatest._

class ModelEvaluationSpec extends FlatSpec with Matchers with DataReader with ModelEvaluation {

  val spark = SparkSession.builder().master("local[2]").appName("test-model-evaluation").getOrCreate()
  val (colNames, trainingData) = readCsv(spark, getClass.getResource("/train.csv").getPath)
  val (colNamesDup, holdOutData) = readCsv(spark, getClass.getResource("/holdout.csv").getPath)

  "The model evaluation" should
    "evaluate the predictive performance of a binary model using holdout data" in {
      val model = new LogisticRegressionWithAdaGrad(spark, colNames, trainingData, holdOutData)
      val evaluation = evaluateBinaryPredictions(model.estimate, model.predict(model.parameterEstimates, holdOutData))
      evaluation.evaluationSet shouldEqual 2101
  }

  it should "evaluate the predictive performance of a linear model using holdout data" in {
    val model = new MLlibLinearRegressionWithSGD(colNames, trainingData, holdOutData)
    val evaluation = evaluateLinearPredictions(model.estimate, model.predict(model.model, holdOutData))
    evaluation.evaluationSet shouldEqual 2101
  }

  it should "be able to produce a summary report with diagnostic data on the model run" in {
    val model = new LogisticRegressionWithAdaGrad(spark, colNames, trainingData, holdOutData)
    val evaluation = model.evaluate
    evaluation.logSummary()
  }

}
