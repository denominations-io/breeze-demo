import org.apache.spark.sql._

import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.mllib.classification._

import breeze.util.SerializableLogging

class MLlibLogisticRegressionWithLBFGS(spark: SparkSession, colNames: Array[String], training: Dataset[LabeledPoint], holdout: Dataset[LabeledPoint]) extends Regression
  with ModelEvaluation
  with SerializableLogging {

  override def description = "Logistic regression with LBFGS"

  import training.sparkSession.implicits._

  val trainingRDD = training.map { lp =>
    org.apache.spark.mllib.regression.LabeledPoint(lp.label, org.apache.spark.mllib.linalg.Vectors.fromML(lp.features)) }
    .rdd

  val regression = new LogisticRegressionWithLBFGS().setNumClasses(2).run(trainingRDD)
  val parameterEstimates = regression.weights

  override def estimate: ModelSummary = {
    val modelFit = ModelFit("num features", regression.numFeatures)
    val coefficients = colNames.zipWithIndex.map { variable => Coefficient(variable._1, parameterEstimates(variable._2)) }
    ModelSummary(description, modelFit, coefficients)
  }

  val predict: (LogisticRegressionModel, Dataset[LabeledPoint]) => Dataset[Prediction] = {
    (model, holdOut) =>
      holdOut.map { lp =>
        val prediction = model.predict(org.apache.spark.mllib.linalg.Vectors.fromML(lp.features))
        Prediction(observed = lp.label, predicted = prediction)
      }
  }

  def evaluate = evaluateBinaryPredictions(estimate, predict(regression, holdout))
}
