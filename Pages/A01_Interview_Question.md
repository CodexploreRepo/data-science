# Data Scientist - Interview Questions

## Table of contents
- [1. SQL Questions](#1-sql-questions)




## 1. SQL Questions
#### SQL#1: Facebook
```SQL
Given the following data:

Table:
searches
Columns:
date STRING date of the search,
search_id INT the unique identifier of each search,
user_id INT the unique identifier of the searcher,
age_group STRING ('<30', '30-50', '50+'),
search_query STRING the text of the search query

Sample Rows:
date | search_id | user_id | age_group | search_query
--------------------------------------------------------------------
'2020-01-01' | 101 | 9991 | '<30' | 'justin bieber'
'2020-01-01' | 102 | 9991 | '<30' | 'menlo park'
'2020-01-01' | 103 | 5555 | '30-50' | 'john'
'2020-01-01' | 104 | 1234 | '50+' | 'funny cats'


Table:
search_results
Columns:
date STRING date of the search action,
search_id INT the unique identifier of each search,
result_id INT the unique identifier of the result,
result_type STRING (page, event, group, person, post, etc.),
clicked BOOLEAN did the user click on the result?

Sample Rows:
date | search_id | result_id | result_type | clicked
--------------------------------------------------------------------
'2020-01-01' | 101 | 1001 | 'page' | TRUE
'2020-01-01' | 101 | 1002 | 'event' | FALSE
'2020-01-01' | 101 | 1003 | 'event' | FALSE
'2020-01-01' | 101 | 1004 | 'group' | FALSE


Over the last 7 days, how many users made more than 10 searches?

You notice that the number of users that clicked on a search result
about a Facebook Event increased 10% week-over-week. How would you
investigate? How do you decide if this is a good thing or a bad thing?

The Events team wants to up-rank Events such that they show up higher
in Search. How would you determine if this is a good idea or not?
```
[(Back to top)](#table-of-contents)
