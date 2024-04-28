## JOIN operator

A `JOIN` clause is used to combine rows from two or more tables, based on a related column between them.

e.g.1

Find all branches and name of them in the table `branches` and `employees`.

```sql
SELECT branches.name, employees.first_name
FROM branches
JOIN employees
ON branches.id = employees.branch_id;
```

e.g.2

<!-- import img --> 
<img src="https://github.com/Laudarisd/Data-science-study/tree/master/src/sql/img/1.png" alt="">


```sql
SELECT Orders.OrderID, Customers.CustomerName, Orders.OrderDate
FROM Orders
INNER JOIN Customers ON Orders.CustomerID=Customers.CustomerID;
```

This will produce the following result:

<img src="https://github.com/Laudarisd/Data-science-study/tree/master/src/sql/img/2.png" alt="">


=================================================================================

### Types of JOINs

<ui> INNER JOIN : Returns records that have matching values in both tables </ui>
<ui> LEFT JOIN : Returns records from the left table, and the matched records from the right table. All the information from left table will be added </ui>
<ui> RIGHT JOIN : Returns records from the right table, and the matched records from the left table. All the information from right table will be added </ui>
<ui> FULL OUTER JOIN : Returns all the records from both tables, and the matched records from the left table. All the information from left table will be added </ui>
<ui> CROSS JOIN : Returns all the records from both tables </ui>


=================================================================================

### References
<li> <a href = "https://www.mysqltutorial.org/mysql-update-data.aspx"> ref 1 </a> </li>
<li> <a href = "https://www.w3schools.com/sql/sql_dates.asp"> ref 2 </a> </li>