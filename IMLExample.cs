using System;
using System.Collections.Generic;
using System.Text;

namespace ml
{
    public interface IMLExample
    {
        void Try();
        string Description { get; }
    }
}
