using Microsoft.ML.Data;

namespace ml
{
    class Prediction
    {
        [ColumnName("Score")]
        public float Price { get; set; }
    }
}
