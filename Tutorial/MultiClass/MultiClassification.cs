using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML;
using static Microsoft.ML.DataOperationsCatalog;

namespace ml.Tutorial.MultiClass
{
    class MultiClassification : MLExample
    {
        public MultiClassification() :base("Многоклассовая")
        {

        }
        public override void Try()
        {
            TrainTestData splitDataView = LoadData();
            ITransformer model = BuildAndTrainModel(splitDataView.TrainSet);
            Evaluate(model, splitDataView.TestSet);
            UseModel(model);
        }

        protected override ITransformer BuildAndTrainModel(IDataView trainSet)
        {


            return null;
        }

        protected override void Evaluate(ITransformer model, IDataView testSet)
        {

        }

        protected override DataOperationsCatalog.TrainTestData LoadData()
        {


            return default;

        }

        protected override void UseModel(ITransformer model)
        {
         
        }
    }
}
