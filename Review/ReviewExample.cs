using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Text;

namespace ml.Review
{
    class ReviewExample : MLExample
    {
        public ReviewExample() : base("Обзор")
        {
        }

        public override void Try()
        {
            HouseData[] houseData = {
               new HouseData() { Size = 1.1F, Price = 1.2F },
               new HouseData() { Size = 1.9F, Price = 2.3F },
               new HouseData() { Size = 2.8F, Price = 3.0F },
               new HouseData() { Size = 3.4F, Price = 3.7F } };
            IDataView trainigData = context.Data.LoadFromEnumerable(houseData);

            var pipeline = context
                                .Transforms.Concatenate("Features", new[] { "Size" })
                                .Append(context.Regression.Trainers.Sdca(labelColumnName: "Price", maximumNumberOfIterations: 100));

            var model = pipeline.Fit(trainigData);

            var size = new HouseData { Size = 2.5F };
            var price = context.Model.CreatePredictionEngine<HouseData, Prediction>(model).Predict(size);

            Console.WriteLine($"Predicted price for size: {size.Size} => {price.Price}");

            HouseData[] testHouseData =
            {
                new HouseData() { Size = 1.1F, Price = 0.98F },
                new HouseData() { Size = 1.9F, Price = 2.1F },
                new HouseData() { Size = 2.8F, Price = 2.9F },
                new HouseData() { Size = 3.4F, Price = 3.6F }
            };

            var testHouseDataView = context.Data.LoadFromEnumerable(testHouseData);
            var testPriceDataView = model.Transform(testHouseDataView);

            var metrics = context.Regression.Evaluate(testPriceDataView, labelColumnName: "Price");

            Console.WriteLine($"R^2: {metrics.RSquared:0.##}");
            Console.WriteLine($"RMS error: {metrics.RootMeanSquaredError:0.##}");
        }

        protected override ITransformer BuildAndTrainModel(IDataView trainSet)
        {
            throw new NotImplementedException();
        }

        protected override void Evaluate(ITransformer model, IDataView testSet)
        {
            throw new NotImplementedException();
        }

        protected override DataOperationsCatalog.TrainTestData LoadData()
        {
            throw new NotImplementedException();
        }

        protected override void UseModel(ITransformer model)
        {
            throw new NotImplementedException();
        }
    }
}
