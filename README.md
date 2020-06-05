# iService
User-Oriented and Privacy-Aware Proximity Services

For the PhD thesis "Edge-Driven Proximity Service Platform for the Internet of Things in Indoor Environments" of Michael Haus

This repository addresses the RQ3 of the research problem:<br/>
**How to realize user-oriented and privacy-aware IoT services by utilizing physical surrounding awareness in the managed indoor IoT areas?**

We address this question by realizing a set of different proximity services. Our fine-granular seamless device association uses the
similarity of visible light signaling in different spatial areas which are impossible to differentiate with Wi-Fi. On this basis, we
can enable social applications, such as to retrieve knowledge from locals, e.g., Alice is a tourist, rides on the subway and wants
to ask locals for the best way to the museum. Moreover, we improved the user privacy in public spaces by turning around the action
cycle for service discovery to be entirely initiated from the IoT environment. The users remain passive and cannot be tracked by
collecting service advertisements via distance-limited VLC when approaching a service area. Furthermore, we automate the authorization
for the remote control of smart homes. A smart home incorporates a communication network that connects the key electrical appliances
and services, and allows to be remotely controlled, monitored or accessed. We fully automate the key management for remote control
of smart homes to improve the usability of system's security based on out-of-band VLC to exchange secret keys and a challenge-response
mechanism for on-demand access requests.

For instance, as minor use case, the automated wireless authentication achieves a touchless authentication experience in a distance-bounding
manner. We improve the usability by avoiding manual distribution and tedious input of passwords for login. We use LocalVLC for M2M
communication to ease the setup of Wi-Fi networks. Other mechanisms to exchange credentials still require human interaction, such as
WPS or QR codes. Target devices include common Wi-Fi equipped devices, e.g., smartphone, tablet, laptop, as well as IoT devices like sensor
boards. The video: https://www.youtube.com/watch?v=e8kjfDNmVSA shows our working demo. Via LocalVLC we broadcast the network name (SSID) and
continuously generated time-based one-time passwords (TOTPs) for the wireless login. The user's end-device retrieves the VLC transmitted
login data and continually scans for nearby wireless networks. In case of spotting a matching SSID, it performs the automated authentication
without any manual interaction.