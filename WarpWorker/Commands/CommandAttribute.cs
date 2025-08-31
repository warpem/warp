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

internal class MockCommandAttribute : Attribute
{
    public string Name;
    
    public MockCommandAttribute(string name)
    {
        Name = name;
    }
}