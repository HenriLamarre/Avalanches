from sunpy.net import Fido, attrs as a

tstart = "2000/01/01"
tend = "2021/05/10"
result = Fido.search(a.Time(tstart, tend),
                     a.hek.EventType("FL"),
                     a.hek.FRM.Name == "SWPC",
                     # a.hek.OBS.Observatory == "GOES"
                     a.hek.FL.GOESCls >= "B1.0")

new_table = result["hek"]["event_starttime", "event_peaktime",
                          "event_endtime", "fl_goescls", "ar_noaanum"]
new_table.write("solar_flares_testnoA.csv", format="csv", overwrite = True)
