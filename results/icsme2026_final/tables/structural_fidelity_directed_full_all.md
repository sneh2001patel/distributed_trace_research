# Structural Fidelity Comparison

All scores are similarities — higher is better.

- **KS sim** = `1 − KS statistic` (0 = maximally different distributions, 1 = identical)
- **MMD sim** = `exp(−MMD)` over 11 graph-level features (node/edge counts, density, degree stats, duration stats, service/op counts)
- **JS sim** = `1 − Jensen-Shannon divergence` over discrete label distributions
- **Valid** = fraction of synthetic (service, op) pairs that appear in the real data
- **Structure Mean** = mean of 7 topology metrics (6 KS sims + MMD sim)
- **Duration Mean** = mean of 2 duration KS sims (per-graph mean duration, all node durations)
- **Semantic Mean** = mean of 4 label metrics (Svc JS, Op JS, SvcOp JS, Valid)
- **Overall** = mean(Structure Mean, Duration Mean, Semantic Mean) — three equal groups

## TT

<table>
<thead>
<tr>
  <th rowspan="2">Generator</th>
  <th rowspan="2">Class</th>
  <th colspan="8">Structure</th>
  <th colspan="3">Duration</th>
  <th colspan="5">Semantic</th>
  <th rowspan="2">Overall</th>
</tr>
<tr>
  <th>Node KS</th>
  <th>Edge KS</th>
  <th>Density KS</th>
  <th>AvgDeg KS</th>
  <th>MaxDeg KS</th>
  <th>AllDeg KS</th>
  <th>MMD Sim</th>
  <th>Mean</th>
  <th>DurMean KS</th>
  <th>AllDur KS</th>
  <th>Mean</th>
  <th>Svc JS</th>
  <th>Op JS</th>
  <th>SvcOp JS</th>
  <th>Valid</th>
  <th>Mean</th>
</tr>
</thead>
<tbody>
<tr>
  <td>random_sampling</td>
  <td>normal</td>
  <td>0.991</td>
  <td>0.580</td>
  <td>0.400</td>
  <td>0.400</td>
  <td>0.479</td>
  <td>0.337</td>
  <td>0.782</td>
  <td><b>0.567</b></td>
  <td>0.656</td>
  <td>0.977</td>
  <td><b>0.816</b></td>
  <td>0.674</td>
  <td>0.433</td>
  <td>0.115</td>
  <td>0.094</td>
  <td><b>0.329</b></td>
  <td><b>0.571</b></td>
</tr>
<tr>
  <td>random_sampling</td>
  <td>abnormal</td>
  <td>0.965</td>
  <td>0.586</td>
  <td>0.394</td>
  <td>0.397</td>
  <td>0.488</td>
  <td>0.353</td>
  <td>0.790</td>
  <td><b>0.567</b></td>
  <td>0.707</td>
  <td>0.971</td>
  <td><b>0.839</b></td>
  <td>0.669</td>
  <td>0.447</td>
  <td>0.112</td>
  <td>0.091</td>
  <td><b>0.330</b></td>
  <td><b>0.579</b></td>
</tr>
<tr>
  <td>random_sampling</td>
  <td>combined</td>
  <td>0.978</td>
  <td>0.583</td>
  <td>0.397</td>
  <td>0.398</td>
  <td>0.484</td>
  <td>0.345</td>
  <td>0.794</td>
  <td><b>0.568</b></td>
  <td>0.682</td>
  <td>0.975</td>
  <td><b>0.829</b></td>
  <td>0.672</td>
  <td>0.440</td>
  <td>0.113</td>
  <td>0.092</td>
  <td><b>0.330</b></td>
  <td><b>0.575</b></td>
</tr>
<tr>
  <td>flat_vae</td>
  <td>normal</td>
  <td>0.988</td>
  <td>0.531</td>
  <td>0.260</td>
  <td>0.230</td>
  <td>0.489</td>
  <td>0.629</td>
  <td>0.635</td>
  <td><b>0.537</b></td>
  <td>0.127</td>
  <td>0.682</td>
  <td><b>0.405</b></td>
  <td>0.558</td>
  <td>0.552</td>
  <td>0.504</td>
  <td>0.760</td>
  <td><b>0.593</b></td>
  <td><b>0.512</b></td>
</tr>
<tr>
  <td>flat_vae</td>
  <td>abnormal</td>
  <td>0.972</td>
  <td>0.599</td>
  <td>0.553</td>
  <td>0.419</td>
  <td>0.577</td>
  <td>0.744</td>
  <td>0.719</td>
  <td><b>0.655</b></td>
  <td>0.137</td>
  <td>0.643</td>
  <td><b>0.390</b></td>
  <td>0.641</td>
  <td>0.629</td>
  <td>0.581</td>
  <td>0.818</td>
  <td><b>0.667</b></td>
  <td><b>0.571</b></td>
</tr>
<tr>
  <td>flat_vae</td>
  <td>combined</td>
  <td>0.986</td>
  <td>0.565</td>
  <td>0.425</td>
  <td>0.325</td>
  <td>0.533</td>
  <td>0.685</td>
  <td>0.689</td>
  <td><b>0.601</b></td>
  <td>0.132</td>
  <td>0.668</td>
  <td><b>0.400</b></td>
  <td>0.601</td>
  <td>0.593</td>
  <td>0.546</td>
  <td>0.788</td>
  <td><b>0.632</b></td>
  <td><b>0.545</b></td>
</tr>
<tr>
  <td>hierarchical_vae</td>
  <td>normal</td>
  <td>0.988</td>
  <td>0.988</td>
  <td>0.988</td>
  <td>0.990</td>
  <td>0.991</td>
  <td>0.956</td>
  <td>0.987</td>
  <td><b>0.984</b></td>
  <td>0.907</td>
  <td>0.979</td>
  <td><b>0.943</b></td>
  <td>0.986</td>
  <td>0.982</td>
  <td>0.981</td>
  <td>1.000</td>
  <td><b>0.987</b></td>
  <td><b>0.971</b></td>
</tr>
<tr>
  <td>hierarchical_vae</td>
  <td>abnormal</td>
  <td>0.972</td>
  <td>0.972</td>
  <td>0.972</td>
  <td>0.972</td>
  <td>0.977</td>
  <td>0.962</td>
  <td>0.990</td>
  <td><b>0.974</b></td>
  <td>0.911</td>
  <td>0.973</td>
  <td><b>0.942</b></td>
  <td>0.973</td>
  <td>0.974</td>
  <td>0.970</td>
  <td>1.000</td>
  <td><b>0.979</b></td>
  <td><b>0.965</b></td>
</tr>
<tr>
  <td>hierarchical_vae</td>
  <td>combined</td>
  <td>0.986</td>
  <td>0.986</td>
  <td>0.986</td>
  <td>0.986</td>
  <td>0.987</td>
  <td>0.959</td>
  <td>0.986</td>
  <td><b>0.982</b></td>
  <td>0.927</td>
  <td>0.977</td>
  <td><b>0.952</b></td>
  <td>0.984</td>
  <td>0.981</td>
  <td>0.980</td>
  <td>1.000</td>
  <td><b>0.986</b></td>
  <td><b>0.974</b></td>
</tr>
</tbody>
</table>

## SN

<table>
<thead>
<tr>
  <th rowspan="2">Generator</th>
  <th rowspan="2">Class</th>
  <th colspan="8">Structure</th>
  <th colspan="3">Duration</th>
  <th colspan="5">Semantic</th>
  <th rowspan="2">Overall</th>
</tr>
<tr>
  <th>Node KS</th>
  <th>Edge KS</th>
  <th>Density KS</th>
  <th>AvgDeg KS</th>
  <th>MaxDeg KS</th>
  <th>AllDeg KS</th>
  <th>MMD Sim</th>
  <th>Mean</th>
  <th>DurMean KS</th>
  <th>AllDur KS</th>
  <th>Mean</th>
  <th>Svc JS</th>
  <th>Op JS</th>
  <th>SvcOp JS</th>
  <th>Valid</th>
  <th>Mean</th>
</tr>
</thead>
<tbody>
<tr>
  <td>random_sampling</td>
  <td>normal</td>
  <td>0.975</td>
  <td>0.973</td>
  <td>0.683</td>
  <td>0.612</td>
  <td>0.836</td>
  <td>0.657</td>
  <td>0.852</td>
  <td><b>0.798</b></td>
  <td>0.688</td>
  <td>0.975</td>
  <td><b>0.831</b></td>
  <td>0.563</td>
  <td>0.817</td>
  <td>0.103</td>
  <td>0.078</td>
  <td><b>0.390</b></td>
  <td><b>0.673</b></td>
</tr>
<tr>
  <td>random_sampling</td>
  <td>abnormal</td>
  <td>0.957</td>
  <td>0.950</td>
  <td>0.686</td>
  <td>0.569</td>
  <td>0.839</td>
  <td>0.629</td>
  <td>0.859</td>
  <td><b>0.784</b></td>
  <td>0.767</td>
  <td>0.972</td>
  <td><b>0.870</b></td>
  <td>0.536</td>
  <td>0.797</td>
  <td>0.108</td>
  <td>0.084</td>
  <td><b>0.381</b></td>
  <td><b>0.678</b></td>
</tr>
<tr>
  <td>random_sampling</td>
  <td>combined</td>
  <td>0.966</td>
  <td>0.969</td>
  <td>0.684</td>
  <td>0.592</td>
  <td>0.852</td>
  <td>0.643</td>
  <td>0.857</td>
  <td><b>0.795</b></td>
  <td>0.732</td>
  <td>0.974</td>
  <td><b>0.853</b></td>
  <td>0.550</td>
  <td>0.810</td>
  <td>0.106</td>
  <td>0.081</td>
  <td><b>0.387</b></td>
  <td><b>0.678</b></td>
</tr>
<tr>
  <td>flat_vae</td>
  <td>normal</td>
  <td>0.973</td>
  <td>0.444</td>
  <td>0.333</td>
  <td>0.115</td>
  <td>0.495</td>
  <td>0.534</td>
  <td>0.604</td>
  <td><b>0.500</b></td>
  <td>0.229</td>
  <td>0.694</td>
  <td><b>0.461</b></td>
  <td>0.717</td>
  <td>0.554</td>
  <td>0.554</td>
  <td>1.000</td>
  <td><b>0.706</b></td>
  <td><b>0.556</b></td>
</tr>
<tr>
  <td>flat_vae</td>
  <td>abnormal</td>
  <td>0.986</td>
  <td>0.805</td>
  <td>0.724</td>
  <td>0.532</td>
  <td>0.718</td>
  <td>0.841</td>
  <td>0.721</td>
  <td><b>0.761</b></td>
  <td>0.126</td>
  <td>0.613</td>
  <td><b>0.370</b></td>
  <td>0.801</td>
  <td>0.698</td>
  <td>0.698</td>
  <td>1.000</td>
  <td><b>0.800</b></td>
  <td><b>0.643</b></td>
</tr>
<tr>
  <td>flat_vae</td>
  <td>combined</td>
  <td>0.992</td>
  <td>0.625</td>
  <td>0.555</td>
  <td>0.324</td>
  <td>0.629</td>
  <td>0.677</td>
  <td>0.673</td>
  <td><b>0.639</b></td>
  <td>0.184</td>
  <td>0.674</td>
  <td><b>0.429</b></td>
  <td>0.769</td>
  <td>0.639</td>
  <td>0.639</td>
  <td>1.000</td>
  <td><b>0.761</b></td>
  <td><b>0.610</b></td>
</tr>
<tr>
  <td>hierarchical_vae</td>
  <td>normal</td>
  <td>0.973</td>
  <td>0.563</td>
  <td>0.626</td>
  <td>0.332</td>
  <td>0.667</td>
  <td>0.415</td>
  <td>0.802</td>
  <td><b>0.626</b></td>
  <td>0.630</td>
  <td>0.164</td>
  <td><b>0.397</b></td>
  <td>0.417</td>
  <td>0.167</td>
  <td>0.167</td>
  <td>1.000</td>
  <td><b>0.438</b></td>
  <td><b>0.487</b></td>
</tr>
<tr>
  <td>hierarchical_vae</td>
  <td>abnormal</td>
  <td>0.986</td>
  <td>0.559</td>
  <td>0.622</td>
  <td>0.307</td>
  <td>0.585</td>
  <td>0.461</td>
  <td>0.781</td>
  <td><b>0.615</b></td>
  <td>0.488</td>
  <td>0.296</td>
  <td><b>0.392</b></td>
  <td>0.602</td>
  <td>0.287</td>
  <td>0.287</td>
  <td>1.000</td>
  <td><b>0.544</b></td>
  <td><b>0.517</b></td>
</tr>
<tr>
  <td>hierarchical_vae</td>
  <td>combined</td>
  <td>0.992</td>
  <td>0.561</td>
  <td>0.624</td>
  <td>0.320</td>
  <td>0.626</td>
  <td>0.437</td>
  <td>0.808</td>
  <td><b>0.624</b></td>
  <td>0.574</td>
  <td>0.230</td>
  <td><b>0.402</b></td>
  <td>0.540</td>
  <td>0.282</td>
  <td>0.282</td>
  <td>1.000</td>
  <td><b>0.526</b></td>
  <td><b>0.517</b></td>
</tr>
</tbody>
</table>
