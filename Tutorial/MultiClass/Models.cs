using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace ml.Tutorial.MultiClass
{
    public class GitHubIssues
    {
        [LoadColumn(0)]
        public string ID { get; set; }
        [LoadColumn(1)]
        public string Area { get; set; }
        [LoadColumn(2)]
        public string Title { get; set; }
        [LoadColumn(3)]
        public string Description { get; set; }

    }

    public class Prediction
    {
        [ColumnName("PredictedLabel")]
        public string Area { get; set; }
    }
}
