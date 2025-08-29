using System;

namespace WarpWorker;

internal class CommandAttribute : Attribute
{
    public string Name;
    
    public CommandAttribute(string name)
    {
        Name = name;
    }
}