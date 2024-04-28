# FUNCTION #

```sql
CREATE TABLE student_record (
    Name VARCHAR(100), 
    Math INT, 
    English INT, 
    Science INT, 
    History INT
);

-- INSERT INTO student_record values('Raman', 95, 89, 85, 81);
-- INSERT INTO student_record values('Rahul' , 90, 87, 86, 81);
-- INSERT INTO student_record values('Mohit', 90, 85, 86, 81);
-- INSERT INTO student_record values('Saurabh', NULL, NULL, NULL, NULL );


Create Function tbl_Update(S_name Varchar(50), M1 INT, M2 INT, M3 INT, M4 INT)
   RETURNS INT
   DETERMINISTIC
   BEGIN
   UPDATE student_record SET Math = M1, English = M2, Science = M3, History = M4 WHERE Name = S_name;
   RETURN 1;
   END;
```