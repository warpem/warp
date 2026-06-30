using System;
using System.IO;
using System.Threading;
using Xunit;
using Warp.Workers.Queue;

namespace Tests.Workers;

public class HeartbeatTests : IDisposable
{
    private readonly string _dir;

    public HeartbeatTests()
    {
        _dir = Path.Combine(Path.GetTempPath(), "hbtest-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_dir);
    }

    public void Dispose() => Directory.Delete(_dir, true);

    [Fact]
    public void WriteTickIncrementsAndKeepsOnlyLatest()
    {
        var hb = new HeartbeatWriter(_dir, "tick-");
        hb.WriteTick();
        hb.WriteTick();
        hb.WriteTick();

        string[] files = Directory.GetFiles(_dir, "tick-*");
        Assert.Single(files);                       // only latest kept
        Assert.Equal(3, HeartbeatReader.MaxSequence(_dir, "tick-"));
    }

    [Fact]
    public void MaxSequenceReturnsMinusOneWhenNoTicks()
    {
        Assert.Equal(-1, HeartbeatReader.MaxSequence(_dir, "tick-"));
    }

    [Fact]
    public void MonitorReportsStallWhenSequenceDoesNotAdvance()
    {
        var hb = new HeartbeatWriter(_dir, "tick-");
        hb.WriteTick();

        // timeout window of 0 ms means: any elapsed time without advance = stalled
        var mon = new HeartbeatMonitor(_dir, "tick-", timeoutMs: 0);
        mon.Observe();                  // sees seq=1 now
        Thread.Sleep(5);
        Assert.True(mon.IsStalled());   // seq did not advance, window exceeded
    }

    [Fact]
    public void MonitorNotStalledWhenSequenceAdvances()
    {
        var hb = new HeartbeatWriter(_dir, "tick-");
        var mon = new HeartbeatMonitor(_dir, "tick-", timeoutMs: 10_000);

        hb.WriteTick();
        mon.Observe();
        hb.WriteTick();
        mon.Observe();
        Assert.False(mon.IsStalled());
    }

    [Fact]
    public void MonitorHonorsStartupGraceWhenNoTicksYet()
    {
        // No ticks written. With a long grace, not yet stalled.
        var mon = new HeartbeatMonitor(_dir, "tick-", timeoutMs: 0, startupGraceMs: 10_000);
        mon.Observe();
        Assert.False(mon.IsStalled());
    }
}
