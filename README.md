## Breeze demo app

demo app - machine learning in Scala with the Spark MLLib and ScalaNLP Breeze libraries. two different versions of a logistic regression are implemented using MLLib and Breeze.

the Spambase development data used in this project was made available for public use through the UCI Machine Learning Repository.

## Dependencies

* JVM
* SBT
* Spark 2.0.2

## Running the app

Compile the project with `sbt assembly` and run the jar with the following parameters:

```
breeze-demo-assembly-0.1.jar \
    --training <training_data>.csv \
    --holdout <holdout_set>.csv \
    --optimizer adagrad
```
