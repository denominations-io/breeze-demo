
import org.apache.spark.sql._
import org.apache.spark.ml.feature._

trait Regression {
  def estimate: Model
  def evaluation: Dataset[LabeledPoint] => Evaluation
}

case class Estimate(variableName: String, estimate: Double, probability: Double)
case class ModelFit(metric: String, value: Double)

case class Model(
                  name: String,
                  modelFit: Option[ModelFit] = None,
                  parameterEstimates: Option[Array[Estimate]] = None
                )