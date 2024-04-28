## SETUP and Installation

1. Mac M1 -- Microsoft Azure SQL

Lots of application are still not compatiable in M1 chips device. 
So lets give a try to set up `Microsoft Azure SQL edge`.

- Download Azure Data Studio from [here](https://docs.microsoft.com/en-us/sql/azure-data-studio/download-azure-data-studio?view=sql-server-ver15#macos-installation)

- Set up docker. To install docker click [here](https://docs.docker.com/desktop/mac/apple-silicon/)

- Let's pull azure sql edge. In Mac terminal run following command:
```bash
docker pull mcr.microsoft.com/azure-sql-edge
```
Again run follwoing comand to setup pw and port

```bash
docker run --cap-add SYS_PTRACE -e ‘ACCEPT_EULA=1’ -e ‘MSSQL_SA_PASSWORD=reallyStrongPwd123@’ -p 1433:1433 --name SQLServer -d mcr.microsoft.com/azure-sql-edge
```

Once the command is run successfully, go back to your mac docker app where we can see `azure...` container.

Now launch Azure Data Studio and click New Connection for setting up the SQL Connection.

- go to new connection
- then on screen :
     > connection name : Microsoft SQL Server
     > server localhost
     > Athentication type : SQL Login
     > User name: sa
     > password: reallyStrongPwd123@
     
- hit connect 

>=== Voila >====

we can see SQL server running on.....

=============================================================


2. My SQL

To install in mac m1: 
```bash
brew install mysql
```

Start and stop MySQL

```bash
Start MySQL – sudo mysql.server start
Stop MySQL – sudo mysql.server stop
Restart MySQL – sudo mysql.server restart
Check status – sudo mysql.server status
```

To uninstall MySQL
```bash
brew uninstall mysql
```

*** How to reset root password in mysql m1 mac
```bash
Make sure you have Stopped MySQL first (above).
Run the server in safe mode with privilege bypass: sudo mysqld_safe --skip-grant-tables
mysql -u root
UPDATE mysql.user SET authentication_string=null WHERE User='root';
FLUSH PRIVILEGES;
exit;
Then
mysql -u root
ALTER USER 'root'@'localhost' IDENTIFIED WITH caching_sha2_password BY 'yourpasswd';
```

Alternative way:

click [here](https://dev.mysql.com/downloads/file/?id=511481) to download MySQl

After installation, add path to `.bash_profile'  file
```bash
export PATH=/usr/local/mysql/bin:$PATH
```

Update the file
```bahs
$ source ~/.bash_profile
```

To start server

[reference](https://www.positronx.io/how-to-install-mysql-on-mac-configure-mysql-in-terminal/)


```bahs
mysql.server start
```




=================================================
3. Install in Ubuntu


```bash
sudo apt update
```

```bash
 sudo apt install mysql-server
 ```

 ***verify MySQL service status***

```bash
systemctl is-active mysql

# output 

active
```

***Configure MySQL server***
```bash
sudo mysql_secure_installation
```

***Log in to MySQL server***

```bash
sudo mysql
```

If password is set while installing mysql, run this command to sun sql
```bash
$ sudo mysql -u root -p
```
then insert mysql password which was set while installing ...

***run this to start sql without restarting server whenever database is changed***
```bash
FLUSH PRIVILEGES
```

Again try to login into MySQL with the  password 

```bash
mysql -u root -p
```
=======================================
***Allow remote access***

```bash
sudo ufw enable
sudo ufw allow mysql
```

==========================================

## Few errors while installing mysql in ubunut

1. Set password has no significance for user root @ ......

Solution: In terminal : sudo mysql

```bash
$ sudo mysql
```

After that : Type this in running sql server in terminal : 
```bash
> ALTER USER 'root@localhost' IDENTIFIED WITH mysql_native_password by 'your new password';
```


It should solve the mentioned error.

Next run this for complete installation

```bash
$ sudo mysql_secure_installation
```


2. Starting MySQL
.. ERROR! The server quit without updating PID file (/usr/local/mysql/data/AXH4TF401WF.pid).

Solution: ..



3. ERROR 2002 (HY000): Can't connect to local MySQL server through socket '/tmp/mysql.sock' (2)

Solution:





