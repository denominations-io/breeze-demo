
trait Regression {
  def description: String
  def estimate: ModelSummary
  def evaluate: Evaluation
}

case class ModelFit(metric: String, value: Double)
case class Coefficient(variableName: String, estimate: Double, probability: Double)

case class ModelSummary(
                  name: String,
                  modelFit: ModelFit,
                  coefficients: Array[Coefficient]
                )