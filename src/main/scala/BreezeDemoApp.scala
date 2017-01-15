case class AppConfig(dataSet: String = "", algorithm: String = "logit-sgd", evaluationFile: String = "")

object BreezeDemoApp extends DataReader with ModelEvaluation {
  def main(args: Array[String]): Unit = {

    val algorithm: Set[String] = Set("logit-sgd", "logit-adagrad", "linear-sgd")
    val parser = new scopt.OptionParser[AppConfig]("scopt") {
      opt[String]("dataset").action((x, c) => c.copy(dataSet = x))
        .required()
        .validate(x => if (x.endsWith(".csv")) success else failure("Needs to be a .csv file"))
        .text("The file path of the training data should be specified as a string.")
      opt[String]("algorithm").action((x, c) => c.copy(algorithm = x))
        .validate(x => if (algorithm.contains(x)) success else failure(s"Optimizer $x not implemented"))
        .text("The optimizer should be specified as a string")
      opt[String]("evaluation_file").action((x, c) => c.copy(evaluationFile = x))
        .text("The name of the evaluation output file should be specified as a string")
    }

    parser.parse(args, AppConfig()) match {
      case Some(config) => demo(config)
      case None         => System.exit(1)
    }

    import org.apache.spark.SparkConf
    import org.apache.spark.sql.SparkSession

    def demo(app: AppConfig): Unit = {
      val conf = new SparkConf()
        .setAppName("Breeze demo")
        .setMaster("local[2]")
        .set("spark.executor.memory", "2g")
      val spark = SparkSession.builder().config(conf).getOrCreate()

      /** Load in the data and split into training and hold out data. */
      val (colNames, data) = readCsv(spark, app.dataSet)
      val splitData = data.randomSplit(Array(70, 30))
      assert(splitData.length == 2)

      val trainingData = splitData(0)
      val holdoutData = splitData(1)

      // TODO: implement self-learning over model and parameter spaces
      /** Estimate and evaluate the selected model. */
      if (app.algorithm.equals("logit-adagrad")) {
        val logitAdaGrad = new LogisticRegressionWithAdaGrad(spark, colNames, trainingData, holdoutData)
        logitAdaGrad.evaluate.generateSummary(app.evaluationFile)
      } else if(app.algorithm.equals("linear-sgd")) {
        val linearSgd = new MLlibLinearRegressionWithSGD(colNames, trainingData, holdoutData)
        linearSgd.evaluate.generateSummary(app.evaluationFile)
      } else {
        val logitSgd = new MLlibLogisticRegressionWithSGD(colNames, trainingData, holdoutData)
        logitSgd.evaluate.generateSummary(app.evaluationFile)
      }
    }
  }
}
