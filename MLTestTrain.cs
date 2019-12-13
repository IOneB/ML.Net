using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML;

namespace ml
{
    public abstract class MLTestTrain<T> : MLExample
    {
        protected readonly string _trainDataPath;
        protected readonly string _testDataPath;
        private readonly char separatorChar;

        protected MLTestTrain(string description, string trainDataPath, string testDataPath, int? seed = null, char separatorChar = '\t')
            : base(description, seed)
        {
            _trainDataPath = trainDataPath;
            _testDataPath = testDataPath;
            this.separatorChar = separatorChar;
        }

        protected override void LoadData()
        {
            trainData = context.Data.LoadFromTextFile<T>(_trainDataPath, separatorChar: separatorChar, hasHeader: true);
            testData = context.Data.LoadFromTextFile<T>(_testDataPath, separatorChar: separatorChar, hasHeader: true);
        }
    }
}
