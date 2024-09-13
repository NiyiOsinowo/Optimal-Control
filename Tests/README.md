**1. Unit tests**

Unit tests are very low level and close to the source of an application. They consist in testing individual methods and functions of the classes, components, or modules used by your software. Unit tests are generally quite cheap to automate and can run very quickly by a continuous integration server.

**2. Integration tests**
Integration tests verify that different modules or services used by your application work well together. For example, it can be testing the interaction with the database or making sure that microservices work together as expected. These types of tests are more expensive to run as they require multiple parts of the application to be up and running.

**3. Functional tests**
Functional tests focus on the application requirements of the code. Functional tests are performed to check if this module functions as intended.They only verify the output of an action and do not check the intermediate states of the system when performing that action. 

**4. Performance testing**
Performance tests help to measure the reliability, speed, scalability, and responsiveness of an application. It can determine if an application meets performance requirements, locate bottlenecks, measure stability during peak traffic, and more.