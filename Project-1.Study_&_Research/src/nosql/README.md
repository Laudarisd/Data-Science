Install mongodb
======================

```
sudo apt update
sudo apt upgrade
--------------------
sudo apt install mongodb
sudo systemctl status mongodb
```

**MongoDB status checking**

```
sudo systemctl status mongodb
```


**MongoDB start, stop , restart command**


```
sudo systemctl start mongodb
sudo systemctl stop mongodb
sudo systemctl restart mongodb
sudo systemctl enable mongodb
sudo systemctl disable mongodb
```

**Starting MongoDB shell**
```
sudo mongo
```

**Create storage path and link to db**

```
mongod --dbpath ./ --logpath ./
```

- After connecting path, we can start db shell in local pc


===================================================

#### Uninstall MongoDB

```
sudo systemctl stop mongodb
sudo apt purge mongodb
sudo apt autoremove
```




