# Taxonomy & Annotation Reference Guide

This document defines the taxonomy of events, patterns, environment variables, maneuvers, and actors for the autonomous driving video annotation dataset.

---

## 1. Taxonomy of Events

1. **Traffic Control Events**  
   Incidents that involve traffic light changes and traffic sign recognition (e.g., speed limits, stop signs, yield, one way, no entry, etc.).

2. **Vehicle Interaction Events**  
   Incidents that involve cut-ins, cut-outs, lane changes, sudden stops from leading vehicles, intelligent vehicle overtaking, and merging with traffic.

3. **Road and Environment Events**  
   Detection of changes in road conditions (e.g., potholes, speed bumps, wet surfaces, road debris) and sudden weather shifts affecting visibility (e.g., rain, snow, fog).

4. **Obstacle Detection Events**  
   Detection of moving objects (e.g., other vehicles, bicycles, pedestrians, animals) or stationary objects (e.g., parked automobiles, roadwork cones, fallen trees, barriers, debris).

5. **Animal and Pedestrian Events**  
   Detection of animals on or crossing the road, and pedestrians present on sidewalks or entering crosswalks/roadways.

6. **Sudden and Unexpected Events**  
   Detecting and responding to emergency vehicles (e.g., ambulances, fire trucks), crashes, vehicle breakdowns, erratic driving, or unexpected road hazards.

---

## 2. Scenario Event Patterns

1. **Co-linear Pattern**  
   Events that take place along the same trajectory/lane that the ego vehicle is navigating.  
   *Example*: A vehicle directly ahead suddenly slows down, or a pedestrian steps onto a crosswalk directly in the ego vehicle's path.

2. **Cross-linear Pattern**  
   Events that involve intersecting trajectories crossing the ego vehicle's route from the sides or perpendicularly.  
   *Example*: A vehicle cuts into the ego lane from a side street, or a pedestrian crosses the street outside a designated crosswalk.

3. **Causal Pattern**  
   Incidents that directly cause or trigger subsequent actions or reactions by the ego vehicle or nearby traffic.  
   *Example*: A leading vehicle brakes abruptly, causing the ego vehicle to brake to maintain safe headway.

4. **Likelihood Pattern**  
   Events requiring anticipation and prediction of potential hazards based on contextual cues.  
   *Example*: A ball bounces into the street, signaling that a child may run after it, prompting the ego vehicle to proactively reduce speed.

---

## 3. Allowed Values & Enumerations

### Environment
- **Road type**: `Urban`, `Semi-urban`, `Rural`, `Highway`, `Unknown`
- **Intersection**: `true`, `false`
- **Traffic control**: `Traffic signal`, `Stop sign`, `Yield`, `None`, `Unknown`
- **Weather**: `Clear`, `Rain`, `Fog`, `Cloudy`
- **Lighting**: `Day`, `Night`, `Dawn/Dusk`

### Scenario Behavior
- **Event type**:
  - `Traffic Control`
  - `Vehicle Interaction`
  - `Road & Environment`
  - `Obstacle Detection`
  - `Animal & Pedestrian`
  - `Sudden & Unexpected`
- **Event pattern**:
  - `Co-linear`
  - `Cross-linear`
  - `Causal`
  - `Likelihood`
- **Severity**: `Low`, `Medium` (or `Moderate`), `High`
- **Ego maneuver**:
  - `Go straight`
  - `Left Turn`
  - `Right Turn`
  - `Lane change`
  - `Cut-in`
  - `Cut-out`
  - `Overtake`
  - `Emergency brake`
  - `Yield`
  - `Follow`
  - `Stop`
  - `Other`
- **Time window**:
  - `start_sec`: Timestamp string (e.g., `"00:05"`) or float/integer seconds (e.g., `5.0`)
  - `end_sec`: Timestamp string (e.g., `"00:12"`) or float/integer seconds (e.g., `12.0`)

### Actors
- **Class**: e.g. `car`, `pedestrian`, `motorcycle`, `bicycle`, `auto_rickshaw`, `truck`, `bus`, `animal`, etc.
- **Count**: Positive integer (e.g. `1`, `2`, `3`)
- **Relative position**: Relative location with respect to the ego vehicle:
  - `front_right`, `front_left`, `front`
  - `ego_lane`, `left_lane`, `right_lane`
  - `sidewalk_left`, `sidewalk_right`
  - `crossing`, `rear`, `rear_left`, `rear_right`

