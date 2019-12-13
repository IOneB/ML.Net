using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML;

namespace ml.Tutorial.Regression
{
    class Regression : MLTestTrain<TaxiTrip>
    {
        public Regression()
            : base("Прогнозирование цен. Регрессия",
            "Tutorial/regression/taxi-fare-train.csv",
            "Tutorial/regression/taxi-fare-test.csv",
            0, ',')
        { }

        protected override ITransformer BuildAndTrainModel()
        {
            var pipeline = context.Transforms.CopyColumns("Label", "FareAmount")
                .Append(context.Transforms.Categorical.OneHotEncoding(outputColumnName: "VendorIdEncoded", inputColumnName: "VendorId"))
                .Append(context.Transforms.Categorical.OneHotEncoding(outputColumnName: "RateCodeEncoded", inputColumnName: "RateCode"))
                .Append(context.Transforms.Categorical.OneHotEncoding(outputColumnName: "PaymentTypeEncoded", inputColumnName: "PaymentType"))
                .Append(context.Transforms.Concatenate("Features", "VendorIdEncoded", "RateCodeEncoded", "PassengerCount", "TripTime", "TripDistance", "PaymentTypeEncoded"))
                .Append(context.Regression.Trainers.FastTree());
            var model = pipeline.Fit(trainData);

            return model;
        }

        protected override void Evaluate(ITransformer model)
        {
            var prediction = model.Transform(testData);
            var metrics = context.Regression.Evaluate(prediction, "Label", "Score");

            Console.WriteLine();
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Model quality metrics evaluation         ");
            Console.WriteLine($"*------------------------------------------------");
            Console.WriteLine($"*       RSquared Score:      {metrics.RSquared:0.##} => 1");
            Console.WriteLine($"*       Root Mean Squared Error:      {metrics.RootMeanSquaredError:#.##} => min");
        }

        protected override void UseModel(ITransformer model)
        {
            var predFunc = context.Model.CreatePredictionEngine<TaxiTrip, Prediction>(model);

            var taxiTrip = new TaxiTrip
            {
                VendorId = "VTS",
                RateCode = "1",
                PassengerCount = 1,
                TripTime = 1140,
                TripDistance = 3.75f,
                PaymentType = "CRD",
                FareAmount = 0
            };

            var predicted = predFunc.Predict(taxiTrip);
            Console.WriteLine($"**********************************************************************");
            Console.WriteLine($"Predicted fare: {predicted.FareAmount}, actual fare: 15.5");
            Console.WriteLine($"**********************************************************************");
        }
    }
}
