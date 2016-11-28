package logit
package optimizers

import org.scalatest._

import breeze.linalg._
import breeze.optimize._

class AdaGradOptimizerSpec extends FlatSpec with Matchers with AdaGradOptimizer {

  "The AdaGradOptimizer" should
    "optimize parameters using breeze's FirstOrderMinimizer" in {
        val parameters = DenseVector(1.4, 0.1, 4.4, 9.2, 3.7, 1.0)
        val f = new StochasticDiffFunction[DenseVector[Double]] {
              def calculate(x: DenseVector[Double]) = {
                    (sum((x - 3.0) :^ 2.0), (x * 2.0) - 6.0)
              }
              val fullRange = 0 to 100
        }

        // TODO: implement property-based testing
        val optimized = optimizeAGD(f, parameters)
        optimized.length shouldEqual parameters.length
  }

}
