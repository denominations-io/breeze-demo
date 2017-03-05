import breeze.util._

import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.regression._
import org.apache.spark.sql._

class MLlibLinearRegressionWithSGD(colNames: Array[String], training: Dataset[LabeledPoint], holdout: Dataset[LabeledPoint]) extends Regression
  with ModelEvaluation
  with SerializableLogging {

  override def description = "Linear regression with stochastic gradient descent"

  val regression = new LinearRegression().setMaxIter(100).setRegParam(0.4).setTol(1E-10)
  val model = regression.fit(training)

  def estimate = {
    val summary = model.summary
    val modelFit = ModelFit("R2", summary.r2)
    val coefficients =
      colNames.zipWithIndex.map { variable =>
        Coefficient(
          variable._1,
          summary.coefficientStandardErrors(variable._2)
        )
      }
    ModelSummary(description, modelFit, coefficients)
  }

  val predict = { (model: LinearRegressionModel, holdOut: Dataset[LabeledPoint]) =>
    import holdOut.sparkSession.implicits._
    model.transform(holdOut)
         .select("label", "prediction")
         .map { case row: Row =>
           Prediction(row.getDouble(0), row.getDouble(1))
          }
  }

  def evaluate = evaluateLinearPredictions(estimate, predict(model, holdout))
}
