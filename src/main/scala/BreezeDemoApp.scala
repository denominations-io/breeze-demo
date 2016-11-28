import logit.learning.ModelEvaluation
import org.apache.spark.SparkConf
import org.apache.spark.sql._

import logit.models._
import logit.tooling._

case class AppConfig(trainingData: String = "", holdoutSet: String = "", optimizer: String = "sgd")

object BreezeDemoApp extends DataReader with ModelEvaluation {
  def main(args: Array[String]): Unit = {

    val optimizer: Set[String] = Set("sgd", "adagrad")

    val parser = new scopt.OptionParser[AppConfig]("scopt") {
      opt[String]("training").action((x, c) => c.copy(trainingData = x))
        .required()
        .validate(x => if (x.endsWith(".csv")) success else failure("Needs to be a .csv file"))
        .text("The file path of the training data should be specified as a string.")
      opt[String]("holdout").action((x, c) => c.copy(holdoutSet = x))
        .required() // TODO: make passing a holdout set optional
        .validate(x => if (x.endsWith(".csv")) success else failure("Needs to be a .csv file"))
        .text("The file path of the holdout set should be specified as a string.")
      opt[String]("optimizer").action((x, c) => c.copy(optimizer = x))
        .validate(x => if (optimizer.contains(x)) success else failure(s"Optimizer $x not implemented"))
        .text("The optimizer should be specified as a string")
    }

    parser.parse(args, AppConfig()) match {
      case Some(config) => demo(config)
      case None         => System.exit(1)
    }

    def demo(app: AppConfig): Unit = {
      val conf = new SparkConf()
        .setAppName("Breeze demo")
        .setMaster("local[2]")
        .set("spark.executor.memory", "2g")
      val spark = SparkSession.builder().config(conf).getOrCreate()

      // TODO: implement filter that allows the discarding of variables from the data
      val (colNames, trainingData) = readCsv(spark, app.trainingData)
      val (colNamesDup, holdoutData) = readCsv(spark, app.holdoutSet)

      // TODO: implement self learning to allow the app to iterate over all implemented model and parameter spaces
      if (app.optimizer.equals("adagrad")) {
        val customModel = new LogisticRegressionWithAdaGrad(spark, trainingData)
        val predictions = customModel.predict(holdoutData)
        evaluate(spark, predictions, colNames)
      } else {
        val mllibModel = new MLLibLogisticRegressionWithSGD(trainingData)
        val predictions = mllibModel.predict(holdoutData)
        evaluate(spark, predictions, colNames)
      }
    }
  }
}
