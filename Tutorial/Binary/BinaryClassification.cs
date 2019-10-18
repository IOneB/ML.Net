using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Text;
using static Microsoft.ML.DataOperationsCatalog;

namespace ml.Tutorial.Binary
{
    internal class BinaryClassification : MLExample
    {
        const string _dataPath = "Tutorial/binary/yelp_labelled.txt";
        const string _modelPath = "model.zip";

        public BinaryClassification() : base("Двоичная классификация") {}

        public override void Try()
        {
            TrainTestData splitDataView = LoadData();
            ITransformer model = BuildAndTrainModel(splitDataView.TrainSet);
            Evaluate(model, splitDataView.TestSet);
            UseModel(model);
        }

        protected override TrainTestData LoadData()
        {
            var data = context.Data.LoadFromTextFile<SentimentialData>(_dataPath, hasHeader: false);
            return context.Data.TrainTestSplit(data, 0.2);
        }

        protected override ITransformer BuildAndTrainModel(IDataView trainSet)
        {
            var estimator =
                context
                .Transforms.Text
                .FeaturizeText("Features", nameof(SentimentialData.Text))
                .Append(context.BinaryClassification.Trainers.SdcaLogisticRegression("Label", featureColumnName: "Features"));

            var model = estimator.Fit(trainSet);

            return model;
        }

        protected override void Evaluate(ITransformer model, IDataView testSet)
        {
            var prediction = model.Transform(testSet);
            var metrics = context.BinaryClassification.Evaluate(prediction, "Label");

            Console.WriteLine("Model quality");
            Console.WriteLine($"Accuracy: {metrics.Accuracy}");
            Console.WriteLine($"F1Score: {metrics.F1Score}"); // мера баланса между точностью и полнотой
            Console.WriteLine($"Auc: {metrics.AreaUnderRocCurve}"); //уверенность модели в правильности классификации с положительными и отрицательными классами. Чем ближе к 1, тем лучше
        }
        protected override void UseModel(ITransformer model)
        {
            var predictionFunc = context.Model.CreatePredictionEngine<SentimentialData, SentimentialPrediction>(model);
            Console.WriteLine("Input data please");
            while (true)
            {
                var answer = Console.ReadLine();
                if (answer.ToLower() == "exit")
                {
                    break;
                }

                var example = new SentimentialData { Text = answer };
                var result = predictionFunc.Predict(example);

                Console.WriteLine($"Prediction: {result.Prediction}, Probability: {result.Probability}");
            }
        }
    }
}
