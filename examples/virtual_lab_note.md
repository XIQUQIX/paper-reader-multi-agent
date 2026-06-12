# Virtual Lab Summary

Virtual Lab is a multi-agent system for scientific research. It uses a PI agent as a project lead and several scientist agents with specialized identities, such as computational biologist, protein engineer, structural biologist, or immunologist.

The PI agent organizes meetings, assigns tasks, asks agents to debate, integrates expert opinions, and moves the project forward. Human researchers intervene only at high-level decision points, such as confirming the research direction or choosing among candidate designs.

A key design is parallel meetings. The same topic is discussed by multiple independent groups of agents at high temperature, creating diverse proposals. A later integration step uses lower temperature to merge the strongest conclusions and reduce noise.

This design aims to compensate for limited human expert availability and to reduce the conservatism of single-pass LLM reasoning.
