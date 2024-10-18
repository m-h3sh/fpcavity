<p align="center">
  <a href="" rel="noopener">
 <img width=887px height=281px src="https://cfi.iitm.ac.in/assets/Abhiyaan-90806928.png" alt="logo"></a>
</p>

<h3 align="center">Team Abhiyaan</h3>

<p align="center"> My Application for Team Abhiyaan's Software Module
    <br> 
</p>

## Table of Contents

- [Intro](#intro)
- [Task 1](#task1)
- [Task 2](#task2)
- [Task 3](#task3)
- [Task 4](#task4)
- [Task 5](#task5)
- [Task 6](#task6)
- [Task 7](#task7)

# INTRO <a name = "intro"></a>

<b><u>Name</u></b> <br>Mahesh Gondi


<b><u>Roll Number</u></b><br>EP23B037

<b><u>Previous Experience</u></b> <br>
Intern at Panoculon Labs, IITM Research Park <br>
January 2024 - Present <br>
Worked on integrating ML models with YOLO, LLava architectures with Android apps. <br>
Also worked with AWS services (Sagemaker, Lambda).

<b><u>Current POR's</b></u><br>
Music Club DC<br>
Tennis Probables

<b> <u>Why I want to join Team Abhiyaan</u></b> <br>
The work I've done previously has exposed me to Computer Vision and ML, and I would really like to work on these fields more through Abhiyaan.
My main objective behind joining this team is the vast learning opportunities it will give me. What excites me most is working on a project that combines hardware with software. I love working collaboratively and am passionate about the things being done here. I also feel like I will have a lot of fun as part of this team, and would love to meet like minded people.<br>
<br>

# TASK 1 <a name = "task1"></a>


##  Subtask A 

### Topics -
1) <b>start_here</b>  - You_are_welcome!
2) <b>You_are_welcome</b> - He Pasta_way!
3) <b>Pasta_way</b> - She said Bison!
4) <b>Bison</b> - Oops!
5) <b>Oops</b> - To get Rebooted!
6) <b>Rebooted</b> - CHALLENGE COMPLETED! <br><br>

To run -
```
cd ctm_ws;
ros2 run capture_the_msg ctm;
ros2 run capture_the_msg solution;
```

## Subtask B 

### Approach - 
To make the turtle move, we need to publish to the cmd_vel topic. This takes input as Twist which contains the linear and angular velocity of the turtle.
Approach for wall rebounding - I can define the 3 walls as x=0, y=0 and x = 11. If the position of the turtle is any of these, it has to rebound. If it hits the top wall, we should flip the Y velocity, and if it hits the side walls, we should flip the X velocity. It also needs to rebound when it hits the player turtle. So whenever the position of turtle1 = turtle2, its Y velocity should flip.

For the bottom turtle, an easy way to make it play the game is to make it follow the x coordinate of the ball turtle. 

One issue I encountered was that sometimes for steep angles, when the ball turtle would rebound, it would trigger the condition for rebound (since it was still in the coordinate range for rebounding) and keep getting stuck. The way I overcame this issue was by introducing a small time delay for checking the rebound condition. This allows the turtle enough time to get out of the rebound zone.<br><br>

To run - 
```
cd pingpongws;
ros2 run pp pp;
```

### Resources -

https://docs.ros.org/en/foxy/Tutorials/Beginner-Client-Libraries/Writing-A-Simple-Py-Publisher-And-Subscriber.html <br>
https://chat.openai.com/ (For understanding Twist and Pose structures)

# TASK 2 <a name = "task2"></a>

## Approach -

I have created a behavior tree with 3 fallback nodes in a sequence. Each has a condition and an Action node associated with it.
In the behavior_tree.cpp file I have created the tree nodes.

1. Approach Ball - <br>
  a. BallReached - Check if player_1 has reached the ball <br>
  b. MoveToBall - Move player_1 to distance 1 from the ball <br>
2. Pass Ball - <br>
  a. PassReceived - Check if player_2 has received the ball <br>
  b. PassBall - Move ball from player_1 position to player_2 <br>
3. Shoot Goal - <br>
  a. GoalScored - Check if goal is scored <br>
  b. ShootGoal - Move ball to goal position when goalie is on the other side <br><br>

To run -
```
cd task2ws;
ros2 run bt_sim_turtle world_setup;
ros2 run behavior_tree behavior_tree;
```

## Resources -
https://rosdriven-turtlesim.netlify.app/turtlesim-python/#multiple-goals <br>
https://docs.ros.org/en/foxy/Tutorials/Beginner-Client-Libraries/Writing-A-Simple-Cpp-Publisher-And-Subscriber.html <br>
https://www.youtube.com/watch?v=T_Q57-audMk <br>
https://medium.com/@nullbyte.in/behavior-trees-for-ros2-part-1-unlocking-advanced-robotic-decision-making-and-control-7856582fb812 <br>
http://wiki.ros.org/turtlesim/Tutorials/Go%20to%20Goal <br>
https://answers.ros.org/question/189543/subscribe-and-publish-velocity/ <br>
https://youtu.be/KO4S0Lsba6I <br>

# TASK 3 <a name = "task3"></a>

# Subtask A


We first look at the player closest to the ball, and can call this player the leader, now we have to position the other two players relative to this.
The leader’s next position will be as close as it can get to the ball. There are now two cases - Let the leader be A and the other two, B and C.  
1. The leader (A) gets to the ball before the opponents closest bot
2. The leader (A) gets to the ball after the opponent

In the <b>first</b> case, after the leader gets the ball, all three bots start moving to the goal. <br><br>
a) If A has space around it towards the goal, it moves towards goal with the ball, B and C go to positions at a specific distance from the opponent players. <br>
b) If A is about to run into a player, it passes the ball to the closest point to B or C (in their path) which can't be intercepted by the other two players of the opponent's team. <br><br>
We have to position B and C most optimally to retain possession or score a goal. To retain possession, we can position them in a place from which they can receive a direct pass from A (as discussed above). To calculate this exact position we can draw straight lines out from the leader and chose a line that gets the ball closest to goal without hitting an opponent bot. If A cannot find a pass to B, it will try to dodge the oncoming opponent bot.
<br>If there is a chance for a pass to be completed which gets the ball near the D of the opponent’s goal, we strategise to score,  else we again retain possession.

In the <b>second</b> case, we have to position most optimally for defence. Assuming the opponent’s bot gets the ball, we need to position our players to either intercept the pass or to tackle the player. <br><br>

a) A (leader) will start moving towards the opponent's leader and try to tackle <br>
b) We identify paths from opponent's leader to the other two bots - i.e. potential pass paths, and position B and C in those paths to intercept. <br> <br>
Assuming the opponent’s players don’t move, we place our players anywhere on the line joining the opponent’s leader player and their other two players. This will ensure that any pass possible will be intercepted. Now, our leader can go to tackle/follow the opponent’s leader. This takes care of all possible moves the opponent could make.

# Subtask B

## Approach -

I am using matplotlib to make the simulation. I'm plotting 10 random points in the region 0 <= x <= 10 and 0 <= y <= 10. Then we plot BHAI, another random point.

To make the animation - at every frame, I'm calculating the distance betweeen each bot and Bhai. Based on this, moving the bot by a distance (1/sens). This way, each bot will have a constant velocity and move towards the point that Bhai is at.

# TASK 4 <a name = "task4"></a>
# Subtask A

## Approach - 
Get the color of the moving ball, find pixels with that color on the screen. Now, I’ll create a mask around these pixels, and find the center of the mask. I am storing the coordinates of this center in a deque, and will plot it to show the past path of the ball. To calculate velocity - (ds/dt)
And for acceleration - (dv/dt). <br>
To plot the line of the path, we take the coordinates of the center from the deque. And we can calculate the thickness based on the index in the list. For points that have a smaller index, the thickness is bigger and the earlier points have a smaller thickness.
For predicting the position of the ball, I’m taking the most recent 2 points, extending their line segment (green) and then plotting the circle at the end of the line segment. <br>
This method can still be optimized - by taking more than 2 points, averaging them out and plotting a line (though we will have to take care of the case when the ball rebounds) <br>
Since the velocity and path are very jerky - I have to smoothen them out. <br>
### Smoothening Path -
To smoothen the path, I am using my list of coordinates of the center of the ball. At every point, I'm updating the center of the ball with the mean of the centers that are +-delta around it. This makes the path smooth, but also rounds the rebounding points. The intervals from [0, delta] and [L-delta, L] do not get smoothened.
<br><br>Rough Path -<br>
![](/task4/roughpath.png)
<br><br>Smooth Path - <br>
![](/task4/smoothpath.png)


### Smoothening Velocity -
One way to smoothen the velocity graph, is to use regression. We have a plot of the velocity with a lot of noise. So to eliminate the noise - the most accurate way would be to do polynomial regression and select an appropriate degree that gives the least loss and best fit.
<br><br>Original Velocity - <br>
![](/task4/originalvel.png)
<br><br>Smooth Velocity - <br>
![](/task4/smoothvel.png)
<br><br>Acceleration - <br>
![](/task4/originalaccel.png) <br>

## Resources -
https://realpython.com/linear-regression-in-python/ <br>
https://pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/<br>

# Subtask B

## Approach -

We take one frame at a time. At each frame, convert it to a binary image and detect contours, these will be our masks. Use the same method as before to find the centers of mask using moments (only if two circles aren't colliding).
<br><br>
![](/task4/multiballtrack.png)

I did not have time to implement center tracking for each circle, but the method I thought of was - <br>
Add the center coordinates from each mask to a list, and then when multiple masks are intersecting, don't update the centers in the list.<br> After the masks aren't intersecting anymore, update the list by joining a line from the last added center to the current center. This will create a smooth line for each ball even though they are colliding and we can't get the center at that time.
<br>
Now that we have a list of all centers, we can calculate velocity and acceleration like before.

## Resources - 
https://www.youtube.com/watch?v=WEzfqCTeI5E

# TASK 5 <a name = "task5"></a>

## Approach - 
First used the <b>SAM</b> model by META AI to segment everything in the given image. But we need a model that separates roads from “not-roads”! So I need to now train a model with a dataset which already has masks separating roads.
So, I trained a <b>Resnet</b> model with a custom dataset built for Indian Roads. The dataset had images and corresponding masks. Using segmentation_models, I trained the model and used it for inference.

From what I have learnt about segmentation - Resnet is a more efficient architecture for segmentation. It is a CNN based architecture.
The previous CNN based models encountered the Vanishing Gradient issue. While calculating the most optimal weights for the neural networks, we calculate the gradient and then use an algorithm like gradient descent to minimize the loss. In previous architecture, if many layers were added to the model, it would lose the gradient at some point. Resnet overcame this issue and hence is now widely used.

The dataset has 6993 images all matched with the corresponding masks. To create the train - test split - I used scikitlearn’s train_test_split function and created a validation set of 10%.

I am using the <b>Adam optimizer</b> - 
	The main difference is that in regular stochastic gradient descent the learning rate is always fixed, but in adam there is a learning rate maintained for each weight in the network and these change as the learning unfolds.

“model = sm.Unet(backbone, encoder_weights='imagenet')”
In this line, I am setting up the Unet model with the Resnet backbone and initializing the backbone with <b>Imagene</b>> (a project for labeling and classifying images to ~22000 categories) weights. <br>

<b>YOLOv8-SEG</b> <br>
YOLO is a "You Only Look Once" architecture. These models have been trained on the COCO dataset. There are multiple sizes of these models - yolov8s-seg, yolov8n-seg, etc

![alt text](https://miro.medium.com/v2/resize:fit:1256/1*V59Hv1ACwrnNRnYSBayKbg.png)

In my colab notebook, I used the largest model x to be most accurate and to compare with SAM.

## Comparison
<b><u>SAM</b></u> - Inference time = 12.96 seconds <br><br>
![](/task5/samoutput.png)<br><br>
<b><u>Unet</b></u> (self trained on IDD Dataset) - 0.7 loss after 10 epochs<br><br>
My model - https://drive.google.com/file/d/15Gp3td7fa0U9zudD8qUhn7SzFInt3x7s/view?usp=sharing <br><br>
<b><u>YOLO-Seg</b></u> - total inference time = 2.86 seconds (with x model) <br><br>
![](/task5/yoloxoutput.jpeg)<br>
<br>
Table below contains the execution times and parameters of each of the YOLOv8 seg models along with SAM. YOLO-seg can be used to classify and segment the types of obstacles after road is segmented.
<br>

| Model   | SAM    | YOLOv8n | YOLOv8s | YOLOv8m | YOLOv8l | YOLOv8x |
|---------|--------|---------|---------|---------|---------|---------|
| Parameters | 94.7M | 3.4M | 11.8M | 27.3M | 46M | 71.8M |
| Size (MB)  |     358    |     6.8    |     23    |    52.4     |    88.1     |   137
| Time (s) (approx)   |    9-12     |    0.2     |    0.15     |    0.13     |    0.12     |    0.13

Total pre and post process times increase as we go from n to x, but the accuracy also increases.

## Resources -

https://www.superannotate.com/blog/guide-to-semantic-segmentation#training-convolutional-neural-networks-for-semantic-segmentation-the-naive-approach

https://segment-anything.com/

https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/

https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/train-yolov8-instance-segmentation-on-custom-dataset.ipynb#scrollTo=FDbMt_M6PiXb

# TASK 7 <a name = "task7"></a>

## Approach -

First, subscribe to the /obstalces topic to receive information about where to spawn the blocks. To dodge the obstacles, the logic I used is - if the center of the block is less than a threshold (say 200), then the car moves to the extreme right side, and if it is greater, then the car moves to the extreme left side of the window. This way the logic is simple and the car won't hit the obstacle. It is inefficient though, as practically, the car only needs to move enough to <b>just</b> avoid the block instead of moving all the way to the extremes of the window
<br><br>

To run - 
```
cd task7ws;
ros2 run nfs obstacles;
ros2 run nfs autogame;
```

## Resources - 

https://github.com/ScareCrow95/ros2-examples-jmu/blob/master/skibot/skibot/skibot_node.py (to combine pygame with ros2)