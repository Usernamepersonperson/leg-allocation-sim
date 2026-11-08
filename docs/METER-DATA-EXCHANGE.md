# Metered-data exchange message

Swiss market parties exchange interval metering data with SDAT-CH, the national implementation of the ebIX "validated metered data" report. A message carries:

1. a header (document id, creation time, process type, sender, recipient),
2. the metering point identifier and the product (what the values mean),
3. the covered interval (start, end) and the resolution (here 15 minutes) with the unit,
4. an ordered list of observations: sequence number, volume, and a condition code (measured, calculated, estimated, missing).

`leg-allocation-sim export-meter-data` writes one message per consumer for one month, carrying the **community-allocated energy** per 15-minute interval (quality "calculated"). The structure it uses (namespace `urn:leg-sim:meter-data-exchange:1`):

```xml
<MeteredDataReport xmlns="urn:leg-sim:meter-data-exchange:1" version="1">
  <HeaderInformation>
    <InstanceDocument><DocumentID/><Creation/></InstanceDocument>
    <ProcessType/><SenderIdentification/><RecipientIdentification/>
  </HeaderInformation>
  <MeteringData>
    <MeteringPointID/><ProductType/>
    <Interval><StartDateTime/><EndDateTime/></Interval>
    <Resolution><Resolution>PT15M</Resolution><Unit>KWH</Unit></Resolution>
    <Observation><Sequence>1</Sequence><Volume>0.000</Volume><Condition>calculated</Condition></Observation>
    <!-- one Observation per interval -->
  </MeteringData>
</MeteredDataReport>
```

Status: element names are simplified and **not validated against the official SDAT-CH schema**; code lists (process type, product type, condition codes, party identifiers) are placeholders. Use it as a readable reference export; map it to the grid operator's required format before any real exchange. `leg_sim.export.parse_meter_data` reads a message back and is used in the tests (sequence 1..N, volumes sum to the allocated energy, 3-decimal kWh = whole Wh).
