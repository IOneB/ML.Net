using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Text;
using static Microsoft.ML.DataOperationsCatalog;

namespace ml
{
    abstract class MLExample : IMLExample
    {
        public string Description { get; }
        protected MLContext context;

        public MLExample(string description)
        {
            Description = description;
            context = new MLContext();
        }
        protected abstract TrainTestData LoadData();
        protected abstract ITransformer BuildAndTrainModel(IDataView trainSet);
        protected abstract void Evaluate(ITransformer model, IDataView testSet);
        protected abstract void UseModel(ITransformer model);

        public abstract void Try();
    }
}
