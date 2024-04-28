## Median

Median is the middle value of a sorted list of numbers. If the list has an even number of elements, the median is the average of the two middle values.

### Example

```python
>>> median([1, 2, 3, 4, 5])
3
>>> median([1, 2, 3, 4, 5, 6])
3.5
```

### pros

- It is easy to compute and comprehend.
- It is not distorted by outliers/skewed data.[4]
- It can be determined for ratio, interval, and ordinal scale

### cons

- It does not take into account the precise value of each observation and hence does not use all information available in the data.
- Unlike mean, median is not amenable to further mathematical calculation and hence is not used in many statistical tests.
- If we pool the observations of two groups, median of the pooled group cannot be expressed in terms of the individual medians of the pooled groups.

