"""Real-world data analysis for MES (the database connected here: ``DATABASE``)."""

# The real-world database connected here and its release. MES names it where it is chosen (the source toggle), in its
# status and on the results computed from it; what MES does with real-world data is described without it.
DATABASE = "MIMIC-IV"
DATASET = "MIMIC-IV v2.2"
# What is known of each database: where its data come from, how many centers, and the limits every study on it
# shares (the data catalog, a result's limitations)
ABOUT = {"MIMIC-IV": {"where": "Beth Israel Deaconess Medical Center, Boston, 2008–2019",
                      "centers": "one academic center",
                      "limits": "inpatient medications only; ICD codes are assigned at discharge"}}


# The real-world databases MES knows of, in the order they are shown: name, what they hold, and whether this version
# can run studies on them. MIMIC-IV is the one connected; the others are shown as not connected yet.
DATABASES = {"mimic": {"name": "MIMIC-IV", "about": "ICU and hospital stays, Beth Israel Deaconess, Boston, 2008–2019",
                       "supported": True},
             "insight": {"name": "INSIGHT", "about": "INSIGHT Clinical Research Network, New York City",
                         "supported": False}}


def database_of(study=None):
    """The database ``study`` ran on, by name: its dataset without the release ("MIMIC-IV"); the connected one when
    the study does not say."""
    dataset = study.get("dataset") if isinstance(study, dict) else None
    return str(dataset or "").split(" v")[0].strip() or DATABASE


def about(name):
    """``ABOUT`` of database ``name``, every field "" when it is not known."""
    return {"where": "", "centers": "", "limits": "", **ABOUT.get(name, {})}
