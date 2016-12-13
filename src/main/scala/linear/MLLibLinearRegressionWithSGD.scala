import breeze.linalg._
import breeze.util._

import org.apache.spark.ml.regression._
import org.apache.spark.sql._

class MLLibLinearRegressionWithSGD(data: Dataset[org.apache.spark.ml.feature.LabeledPoint]) extends Regression
  with ModelEvaluation
  with SerializableLogging {

  val properties = Model("Linear regression with SGD")

  val regression = new LinearRegression().setMaxIter(100).setRegParam(0.4).setTol(1E-10)
  val model = regression.fit(data)

  override def estimate = { properties }

  override def evaluation = {
    holdOut: Dataset[org.apache.spark.ml.feature.LabeledPoint] =>
      val appliedModel = model.transform(holdOut)

      import holdOut.sparkSession.implicits._
      val predictions =
        appliedModel
          .select("label", "prediction")
          .map { case row: Row =>
            Prediction(row.getDouble(0), row.getDouble(1))
          }

      evaluateLinearModel(estimate, predictions)
  }

}
