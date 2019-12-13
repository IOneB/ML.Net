using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Text;
using static Microsoft.ML.DataOperationsCatalog;

namespace ml
{
    public abstract class MLExample : IMLExample
    {
        public string Description { get; }
        protected static MLContext context;
        protected IDataView testData;
        protected IDataView trainData;

        public MLExample(string description, int? seed = null)
        {
            Description = description;
            context = new MLContext(seed);
        }
        protected abstract void LoadData();
        protected abstract ITransformer BuildAndTrainModel();
        protected abstract void Evaluate(ITransformer model);
        protected abstract void UseModel(ITransformer model);

        public virtual void Try()
        {
            LoadData();
            ITransformer model = BuildAndTrainModel();
            Evaluate(model);
            UseModel(model);
        }
    }
}
