# Kinematic AR: Modulating Virtual Production Elements via Embodied Movement Qualities
---

## Main Reference Review:
**Caramiaux et al. (2015) "Towards a Movement Quality Framework for Interactive Systems" (MOCO)**
This paper presents a theoretical and computational framework for capturing and utilizing "Movement Qualities" rather than raw biomechanical kinematics. Philosophically, it aligns with Embodied Interaction by treating the human body not just as a geometric cursor, but as an expressive entity. Its relevance to this project lies in its methodology for abstracting raw sensor data (which in my case is the ZED camera's 3D skeletal tracking) into higher-level descriptors like effort, speed, and fluidity. By applying this philosophy, my Virtual Production hybrid system shifts from a tool that merely "sees" where an actor is, to an embodied interface that "feels" how an actor moves, allowing the Augmented Reality environment to react contextually to human expression.

## Implementation Idea & Description
In a standard Virtual Production LED volume, interacting with foreground AR elements often feels rigid. Using the ZED camera's skeletal tracking (via Blueprint/C++ in Unreal Engine 5), I am capturing the real-time joint data of the user.
To adhere to the EI principle of avoiding direct 1:1 mapping, the implementation calculates specific movement descriptors:

### Velocity & Acceleration (Time/Effort):
Calculating the delta of joint positions over time to determine if a movement is "Sudden" or "Sustained."

### Spatial Expansiveness (Space):
Calculating the bounding volume of the tracked skeleton to determine if the posture is contracted or expanded.

These descriptors are then piped into Unreal Engine's Niagara particle system or AR physics volume. High-energy, sudden movements repel or agitate the virtual objects, while sustained, slow movements attract or calm them, establishing a continuous action-perception loop between the physical body and the hybrid XR environment.

## Code Snippets:
(Placeholder).

## Additional References:
Loke, L., & Robertson, T. (2013). Moving and making methodologies in HCI: The Feldenkrais framework. ACM Transactions on Computer-Human Interaction (TOCHI).

Stereolabs ZED SDK Documentation (Skeletal Tracking API).