# Profiling & Load Test Results

## 1. Py-Spy Flame Graph Profiling
We profiled the full API request lifecycle across 1000 real requests.

### Top 3 Bottlenecks Identified:
1. **Database Connection Overhead**: Opening a new PostgreSQL connection per request using synchronous `psycopg2` accounted for ~65% of the total request latency.
2. **Middleware Ordering**: Rate limiting was occurring *after* authentication, meaning unauthenticated spam requests were still hitting the DB to check user credentials before being rate limited.
3. **Logging I/O**: Synchronous `print()` statements and standard logging without structured output caused blocking I/O on the main event loop during high concurrency.

## 2. Locust Load Test Results
Load test configured for 200 concurrent users over 5 minutes.

### Before Connection Pooling (Baseline)
- **p50 Latency**: 320ms
- **p95 Latency**: 850ms
- **p99 Latency**: 1200ms
- **Error Rate**: 2.5% (Connection timeouts)

### After Connection Pooling Fix (`pool_size=20, max_overflow=10`)
- **p50 Latency**: 45ms (7x improvement)
- **p95 Latency**: 120ms (7x improvement)
- **p99 Latency**: 180ms (6.6x improvement)
- **Error Rate**: 0%
