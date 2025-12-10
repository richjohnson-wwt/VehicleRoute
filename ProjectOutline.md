From Product Owner
-------------
Great meeting today! If you have any questions, please let me know.  If we could touch base every couple of days that would be great.
 
Below are the details we discussed and attached,  you will find.
 
PCRefreshLocations – is the spreadsheet that our Field Services team is using at this point to help them determine teams and schedule
PNCSiteList_Maptest – there are two tabs from two different projects that have addresses of each location we worked out – PNC Workstation Project and PNC Phone
Ascension Health Site List – Not as large list of sites
 
To help quickly determine a Schedule and Team’s routes for a MultiSite Project
 
From the Main Software – Mapping/Routing Tool

| Feature | Description |
| --- | --- |
| Map Site Locations | Ability to pin and label multiple site locations on a map |
| Import Data | Upload Site Data Via CSV, Excel, etc |
| Route Optimization | Automatically Generate optimal routes |
| Custom Parameters | Input project dates, working hours and daily site goals |

Custom Parameters:
* Team Assignments – allocate routes based on number of teams
    Auto-calculate number of teams needed
    Or divide based on regions and/or number of sites
* Time Estimation – estimate time per site and total project duration
    Set Project Start/End Dates
    Define working hours per day
    Set number of sites per day per team
    Set the run rate across the regions that can be completed per each bridge/cut
    Set hours to complete each site; also by t shirt size
    Load in breaks; fire, holidays, blackout dates
        Can a region be completed before a break comes or push to start after the break
        Typical Weather patterns thought about; ie. outside work prob shouldn’t be done in MN during the winter; factor in starting Northern sites and work way south
* Routing –
    Team radius based on region – not to go over x amount of miles (avoidance of overnight travel)
* Crawl, Walk, Run – Ramp Phase approach
 

Project Outline
-------------------

VRPTW (Vehicle Routing Problem with Time Windows)

Recommended PoC Architecture

1. Data Import & Geocoding

Take CSV → clean → geocode addresses

* Use Nominatim (OpenStreetMap) if you want totally free API
* Or Google Maps if accuracy/time windows matter
* Store results as lat/long and optionally timezone if dates matter later

2. Build a Travel Time Matrix

You need realistic driving times, not Euclidean distance.

* Use an API like OSRM, OpenRouteService, or Google Directions

* Cache results locally so repeated runs are fast

3. Define Scheduling Constraints for each site:

* service_time
* time_window (start/end dates → convert to earliest/latest arrival)
* Optional: preferred day buckets

4. Routing Solver

For the spike: Use Google OR-Tools — it's perfect for VRP/VRPTW.

5. Output & visualization

Once OR-Tools give you routes:

* Display assignments
* Day schedule showing arrival times
* Plot paths using networkx + matplotlib 

Even better:

* Use OSRM tile route snapshots to draw true road paths


Minimal Spike Plan (1–2 weeks total)
| Step    | Deliverable                        | Tools                                 |
| ------- | ---------------------------------- | ------------------------------------- |
| Day 1–2 | CSV ingest & geocoding             | pandas, geopy/Nominatim               |
| Day 3–5 | Travel matrix generation           | OSRM or Google API                    |
| Day 5–7 | Basic VRPTW solver                 | OR-Tools (CP-SAT or RoutingModel)     |
| Day 7–9 | Visualization & team count summary | networkx, matplotlib/folium           |
| Day 10+ | Parameter experimentation          | sliders/toggles for dates & crew size |


What you show the business

* “Here’s your 50 sites. With constraints, you need 5 teams.”

* “Here’s the optimal route for each team.”

* “Here’s how total time changes if we add crew/hours/flex.”

Even a slightly rough route is enough for stakeholders to validate the direction.

Stretch Ideas (if time)

* Integrate cluster-first → route-second approach for speed
* Try heuristics like K-Means to form team regions before routing
* UI to upload new CSVs
* Output per-team itinerary to CSV or calendar events