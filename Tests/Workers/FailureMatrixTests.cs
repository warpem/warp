using Xunit;
using Warp.Workers.Scheduling;

namespace Tests.Workers;

public class FailureMatrixTests
{
    [Fact]
    public void BlacklistsHostAfterDistinctTaskThreshold()
    {
        var m = new FailureMatrix(hostBlacklistThreshold: 3, taskPoisonThreshold: 99);
        m.RecordFailure("nodeA", "task1");
        m.RecordFailure("nodeA", "task2");
        Assert.False(m.IsHostBlacklisted("nodeA"));
        m.RecordFailure("nodeA", "task3");
        Assert.True(m.IsHostBlacklisted("nodeA"));
    }

    [Fact]
    public void DuplicateTaskOnSameHostDoesNotInflate()
    {
        var m = new FailureMatrix(hostBlacklistThreshold: 3, taskPoisonThreshold: 99);
        m.RecordFailure("nodeA", "task1");
        m.RecordFailure("nodeA", "task1");
        m.RecordFailure("nodeA", "task1");
        Assert.False(m.IsHostBlacklisted("nodeA")); // only 1 distinct task
    }

    [Fact]
    public void PoisonsTaskAfterDistinctHostThreshold()
    {
        var m = new FailureMatrix(hostBlacklistThreshold: 99, taskPoisonThreshold: 3);
        m.RecordFailure("nodeA", "task1");
        m.RecordFailure("nodeB", "task1");
        Assert.False(m.ShouldPoison("task1"));
        m.RecordFailure("nodeC", "task1");
        Assert.True(m.ShouldPoison("task1"));
    }

    [Fact]
    public void RetryCapAlsoPoisons()
    {
        var m = new FailureMatrix(hostBlacklistThreshold: 99, taskPoisonThreshold: 99, retryCap: 3);
        Assert.False(m.ShouldPoisonByRetry(2));
        Assert.True(m.ShouldPoisonByRetry(3));
        Assert.True(m.ShouldPoisonByRetry(4));
    }
}
