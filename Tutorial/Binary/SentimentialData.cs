using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace ml.Tutorial.Binary
{
    class SentimentialData
    {
        [LoadColumn(0)]
        public string Text { get; set; }
        [LoadColumn(1), ColumnName("Label")]
        public bool Sentiment { get; set; }
    }

    class SentimentialPrediction : SentimentialData
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }
        public float Probability { get; set; }
        public float Score { get; set; }
    }
}
