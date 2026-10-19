"""Build the five-page, vector-based competition graphic from audited results.
Run: python analysis/build_graphic.py --results results --output reports/Escape_the_Feed_Phase_1.pdf
"""
from pathlib import Path
import argparse,json,math
import numpy as np
from matplotlib import get_data_path
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor,Color
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle

ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,default=Path('results'));ap.add_argument('--output',type=Path,default=Path('reports/Escape_the_Feed_Phase_1.pdf'));ap.add_argument('--repo-url',default='');args=ap.parse_args()
R=args.results;s=json.loads((R/'results.json').read_text());e=json.loads((R/'extended_results.json').read_text());tr=np.load(R/'trajectories.npz')
F=s['models']['Full history'];T=s['models']['20-item reset'];B=e['benchmark'];A=e['fairness'];HM=e['heatmap']
fontdir=Path(get_data_path())/'fonts/ttf'
for name,file in [('Body','DejaVuSans.ttf'),('Bold','DejaVuSans-Bold.ttf'),('Italic','DejaVuSans-Oblique.ttf'),('Display','DejaVuSerif.ttf')]:pdfmetrics.registerFont(TTFont(name,str(fontdir/file)))
pdfmetrics.registerFontFamily('Body',normal='Body',bold='Bold',italic='Italic',boldItalic='Bold')
W,H=595.276,841.89;L=42;RW=W-2*L
INK='#142E38';PAPER='#F7F4ED';TEAL='#007E77';MINT='#71D5BE';CORAL='#CB694B';GOLD='#EFC47C';MUTED='#556D72';PALE='#E7EEE8';RULE='#CCD7D1';WHITE='#FFFFFF';LIGHT='#BFD0D0'
args.output.parent.mkdir(parents=True,exist_ok=True)
c=canvas.Canvas(str(args.output),pagesize=(W,H),pageCompression=1)
c.setTitle('Escape the Feed | Your tastes can change. Can your feed?');c.setAuthor('Escape the Feed team');c.setSubject('MovieLens 32M: simulated steering, genre exposure, relevance and standardized group differences')
layouts=[]
def rect(x,y,w,h,color,r=0):
    c.setFillColor(HexColor(color));c.setStrokeColor(HexColor(color))
    if r:c.roundRect(x,H-y-h,w,h,r,stroke=0,fill=1)
    else:c.rect(x,H-y-h,w,h,stroke=0,fill=1)
def text(x,y,s,size=10,font='Body',color=INK,align='left'):
    c.setFillColor(HexColor(color));c.setFont(font,size)
    width=pdfmetrics.stringWidth(str(s),font,size)
    assert (x-width if align=='right' else x)>=17 and (x if align=='right' else x+width)<=W-17,(s,x,width)
    if align=='right':c.drawRightString(x,H-y,str(s))
    elif align=='center':c.drawCentredString(x,H-y,str(s))
    else:c.drawString(x,H-y,str(s))
def para(x,y,w,s,size=10,lead=None,color=INK,maxh=1000,font='Body'):
    p=Paragraph(s,ParagraphStyle('p',fontName=font,fontSize=size,leading=lead or size*1.46,textColor=HexColor(color)))
    _,h=p.wrap(w,1000);assert h<=maxh+.1,(s,h,maxh);assert y+h<790,(s,y,h)
    p.drawOn(c,x,H-y-h);layouts.append({'page':page,'y':y,'bottom':y+h,'text':s});return y+h
def line(x1,y1,x2,y2,color=RULE,width=.7,dash=None):
    c.setStrokeColor(HexColor(color));c.setLineWidth(width);c.setDash(dash or []);c.line(x1,H-y1,x2,H-y2);c.setDash([])
def circle(x,y,r,color):
    c.setFillColor(HexColor(color));c.circle(x,H-y,r,stroke=0,fill=1)
def page_start(n,tag,dark=False):
    global page;page=n;bg=INK if dark else PAPER;fg=WHITE if dark else INK;mute=LIGHT if dark else MUTED
    rect(0,0,W,H,bg);rect(L,31,5,16,MINT if dark else TEAL);text(L+14,43,'ESCAPE THE FEED',10,'Bold',fg)
    text(W-L,42,'BEHIND THE FEED / PHASE 1',7.8,'Bold',mute,'right')
    text(L,77,f'0{n} / {tag.upper()}',8.5,'Bold',MINT if dark else TEAL)
    line(L,797,W-L,797,LIGHT if dark else RULE,.6)
    text(L,816,'MovieLens 32M  /  Offline simulation  /  No demographic inference',6.9,'Body',mute)
    for j in range(5):rect(W-L-64+j*10,809,6,3,(MINT if dark else TEAL) if j==n-1 else (MUTED if dark else RULE),1)
    text(W-L,818,str(n),8,'Bold',fg,'right')
def title(a,b=None,dark=False,size=28):
    fg=WHITE if dark else INK;text(L,116,a,size,'Display',fg)
    if b:text(L,153,b,size,'Display',fg)
def nextline(y,s,dark=False):
    line(L,y-12,W-L,y-12,MUTED if dark else RULE)
    text(L,y+3,'NEXT',7.3,'Bold',MINT if dark else TEAL)
    para(L+42,y-7,RW-42,s,9,13,LIGHT if dark else MUTED,maxh=30)
def pct(v):return f'{v*100:.1f}%'

# 1. A question, a comparison and the complete observed response curve.
page_start(1,'Can a feed follow a new interest?',True)
title('Your tastes can change.','Can your feed?',True,29)
para(L,174,RW,'We tested how one history-based recommender responds when a profile explores an infrequently liked movie genre. The histories are real; the new likes are simulated.',11.1,16,WHITE,maxh=58)
text(L,247,'TARGET REACHED WITHIN FIVE NEW LIKES',8.5,'Bold',LIGHT)
text(L,303,pct(F['cumulative_success'][5]),45,'Bold',WHITE)
text(311,303,pct(T['cumulative_success'][5]),45,'Bold',MINT)
text(L,326,'Full history',12,'Bold',WHITE);text(311,326,'Recent-20 reset',12,'Bold',MINT)
text(L,347,f"{F['success5_count']} of {s['cohort_n']:,} profiles",9,'Body',LIGHT)
text(311,347,f"{T['success5_count']} of {s['cohort_n']:,} profiles",9,'Body',LIGHT)
para(L,363,RW,f"<b>+{100*s['paired_gain5']:.2f} percentage points</b> after a reset. Paired 95% interval: {100*s['paired_gain5_ci'][0]:.2f} to {100*s['paired_gain5_ci'][1]:.2f} points.",10,14.5,WHITE,maxh=35)
rect(27,414,W-54,231,PAPER,13)
text(L,439,'MORE NEW LIKES HELPED, BUT MOST TRIALS STILL FELL SHORT',8.1,'Bold',MUTED)
x0,x1,y0,y1=83,502,600,467
for tick in [0,5,10,15,20]:
    yy=y0-tick/22*(y0-y1);line(x0,yy,x1,yy,RULE,.6);text(x0-10,yy+3,str(tick),8,'Body',MUTED,'right')
text(L,459,'Success (%)',7.5,'Body',MUTED)
for tick in [0,2,4,6,8,10]:text(x0+tick/10*(x1-x0),617,str(tick),8,'Body',MUTED,'center')
text((x0+x1)/2,635,'Simulated new likes added',8,'Body',MUTED,'center')
xx=x0+5/10*(x1-x0);line(xx,y0,xx,y1,MUTED,.8,[3,3])
for vals,color,label,dy in [(F['cumulative_success'],INK,'Full',-9),(T['cumulative_success'],TEAL,'Reset',-9)]:
    pts=[(x0+k/10*(x1-x0),y0-v*100/22*(y0-y1)) for k,v in enumerate(vals)]
    for a,b in zip(pts,pts[1:]):line(*a,*b,color,2.3)
    for k in [5,10]:circle(*pts[k],3.6,color)
    text(pts[5][0]+9,pts[5][1]+dy,pct(vals[5]),8,'Bold',color)
    text(x1+9,pts[10][1]+3,pct(vals[10]),8.5,'Bold',color)
para(L,664,RW,'<b>The test:</b> 2,000 selected profiles received the same ten assigned target-genre likes under both methods. A reset keeps only the latest 20 starting likes, then adds the new ones.',9.7,14.4,WHITE,maxh=45)
para(L,715,RW,'<b>Success:</b> reach at least 3 target-genre movies in the top 10 at least once by that step. Both methods began below 3 by design. This measures a response to new signals, not proof of harm.',9.2,13.5,LIGHT,maxh=45)
c.showPage()

# 2. Show exposure directly, with a complete, labelled heatmap.
page_start(2,'What gets offered?')
title('Past interests shape','the choices in front of you.')
para(L,174,RW,'Before adding any new likes, we grouped steering-test profiles by their dominant historical genre. The full-history model offered each group a different genre mix.',11,16,maxh=52)
shown=[g for g in HM['groups'] if g['n']>=30];keep=np.argsort(HM['overall_exposure'])[::-1][:10];other=[j for j in range(len(HM['genres'])) if j not in keep]
labels=[HM['genres'][j] for j in keep]+['Other'];V=np.array([[g['exposure'][j] for j in keep]+[sum(g['exposure'][j] for j in other)] for g in shown])*100
assert np.allclose(V.sum(axis=1),100)
text(L,241,f"{sum(g['n'] for g in shown):,} of {s['cohort_n']:,} profiles shown  /  Groups with at least 30 profiles",8.3,'Bold',TEAL)
cx=146;cw=(W-L-cx)/len(labels);hy=312;rh=41
for j,label in enumerate(labels):
    c.saveState();c.translate(cx+j*cw+cw/2,H-(hy-12));c.rotate(42);c.setFont('Body',8);c.setFillColor(HexColor(MUTED));c.drawString(0,0,label);c.restoreState()
def shade(v):
    a=np.array([.92,.95,.91]);b=np.array([0,.40,.39]);z=min(v/35,1);q=a*(1-z)+b*z
    return '#'+''.join(f'{int(x*255):02x}' for x in q)
for i,g in enumerate(shown):
    yy=hy+i*rh;text(L,yy+19,g['group'],10,'Bold');text(L,yy+32,f"n = {g['n']:,}",7.8,'Body',MUTED)
    for j,v in enumerate(V[i]):
        rect(cx+j*cw+1,yy+1,cw-2,rh-2,shade(v),3);text(cx+(j+.5)*cw,yy+25,f'{v:.0f}',8.6,'Bold',WHITE if v>22 else INK,'center')
text(L,588,'Mean genre share of the top 10 (%)',8.5,'Bold')
for k in range(70):rect(305+k*2,578,2.05,9,shade(k/69*35))
text(298,599,'0',7.5,'Body',MUTED);text(453,599,'35',7.5,'Body',MUTED,'right')
para(L,615,RW,'Each movie is split equally across its genre labels before averaging. Rows total 100% before rounding; Other pools the remaining genres and any unlabelled share.',9.2,13.3,MUTED,maxh=43)
scifi=next(g for g in HM['groups'] if g['group']=='Sci-Fi')
rect(L,674,RW,65,PALE,8)
para(L+14,686,RW-28,f"<b>A detail worth testing:</b> Sci-Fi-led profiles received {100*scifi['exposure'][HM['genres'].index('Action')]:.1f}% Action versus {100*scifi['exposure'][HM['genres'].index('Sci-Fi')]:.1f}% Sci-Fi. Different exposure can reflect preferences and catalogue supply; it does not establish unfairness by itself.",9.6,14,maxh=45)
nextline(766,'Does personalization trade broader genre exposure for better predictions?');c.showPage()

# 3. Use a different, explicitly identified chronological cohort.
page_start(3,'The trade-off')
title('Easier to redirect.','Not automatically more diverse.')
para(L,174,RW,'A separate chronological test used real later likes, with no simulated signals. Full history predicted later likes best; the reset scored lower on both relevance and genre diversity.',10.7,15.8,maxh=51)
methods=['Popularity','Full history','20-item reset'];cols=[CORAL,INK,TEAL]
def benchmark_panel(y,metric,heading,explain,maxval,ticks,fmt):
    rect(27,y,W-54,178,WHITE,12);text(L,y+25,heading,12,'Bold');text(L,y+44,explain,8.3,'Body',MUTED)
    ax=191;end=490;width=end-ax
    for tick in ticks:
        x=ax+tick/maxval*width;line(x,y+59,x,y+137,RULE,.55)
        text(x,y+158,f'{tick:g}',7.8,'Body',MUTED,'center')
    for i,(name,color) in enumerate(zip(methods,cols)):
        yy=y+72+i*29;v=B['methods'][name][metric];mean=v['mean'];lo,hi=v['ci']
        text(L,yy+3,'Recent-20 reset' if i==2 else name,9.1,'Bold' if i!=0 else 'Body',color)
        rect(ax,yy-6,mean/maxval*width,12,color,3)
        a=ax+lo/maxval*width;b=ax+hi/maxval*width;line(a,yy,b,yy,INK,.85);line(a,yy-4,a,yy+4,INK,.85);line(b,yy-4,b,yy+4,INK,.85)
        text(max(b+9,ax+mean/maxval*width+9),yy+4,format(mean,fmt),10,'Bold',color)
benchmark_panel(242,'ndcg10','Relevance to later likes','nDCG@10, from 0 to 1. Higher means better retrieval of later likes.',.16,[0,.04,.08,.12,.16],'.3f')
benchmark_panel(438,'effective_genres','Genre diversity','Effective genre count. Higher means a broader, more balanced genre mix.',10,[0,2,4,6,8,10],'.2f')
d=B['paired']['Full history minus Popularity'];rect(L,637,RW,80,PALE,8)
para(L+14,650,RW-28,f"<b>Full history versus popularity:</b> relevance rose by {d['ndcg10']['difference']:.3f} (paired 95% interval {d['ndcg10']['ci'][0]:.3f} to {d['ndcg10']['ci'][1]:.3f}), while effective genre count fell by {-d['effective_genres']['difference']:.2f} (interval {-d['effective_genres']['ci'][1]:.2f} to {-d['effective_genres']['ci'][0]:.2f}).",10,15,maxh=56)
para(L,730,RW,f"n = {B['n']:,}; Jan-Oct 2023 later likes. Bars are user means; whiskers are 95% user-bootstrap intervals. Genre labels measure content variety, not diversity of viewpoints. Unrated films are not known dislikes.",8.6,12.5,MUTED,maxh=40)
c.showPage()

# 4. Descriptive subgroup standardization, without overstating uncertainty.
page_start(4,'Who benefits from a reset?')
title('The same reset.','Different opportunities to steer.')
para(L,174,RW,'We compared success within five new likes for profiles with short and long histories, using the same target-genre and starting-exposure weights for both groups.',10.8,16,maxh=51)
text(L,245,'STANDARDIZED SUCCESS RATE (%)',8.6,'Bold',TEAL)
text(L,267,'Short history: 20-50 likes',8.5,'Body',INK);circle(311,264,3.5,TEAL);text(321,267,'Long history: 101+ likes',8.5,'Body',TEAL)
circle(L-10,264,3.5,INK)
ax=183;end=493;yybase=303;ww=end-ax
for tick in [0,3,6,9,12]:
    xx=ax+tick/12*ww;line(xx,306,xx,445,RULE,.7);text(xx,463,str(tick),8,'Body',MUTED,'center')
for i,name in enumerate(['Full history','20-item reset']):
    yy=328+i*83;text(L,yy+9,'Full history' if i==0 else 'Recent-20 reset',10,'Bold')
    for offset,group,color in [(-4,'Short (20-50)',INK),(20,'Long (101+)',TEAL)]:
        v=A['methods'][name]['groups'][group]['rate']*100;y=yy+offset
        if v:rect(ax,y-7,v/12*ww,14,color,3)
        else:circle(ax,y,2.7,color)
        text(ax+v/12*ww+9,y+3,f'{v:.1f}%',10,'Bold',color)
line(L,491,W-L,491)
text(L,516,'GAP: SHORT MINUS LONG',8.6,'Bold',MUTED)
text(L,557,f"{A['methods']['Full history']['signed_gap']*100:+.1f} pp",28,'Bold',INK)
text(315,557,f"{A['methods']['20-item reset']['signed_gap']*100:+.1f} pp",28,'Bold',TEAL)
text(L,578,'Full history',10,'Body',MUTED);text(315,578,'Recent-20 reset',10,'Body',MUTED)
para(L,600,RW,'Both groups became more responsive, but the absolute gap grew. A zero gap would mean equal rates on this measure; it could also mean equally poor responsiveness.',10.5,15.5,maxh=48)
text(L,650,'Gaps use unrounded rates; displayed percentages are rounded.',8.1,'Body',MUTED)
rect(L,661,RW,96,PALE,9)
para(L+14,673,RW-28,f"<b>How we made tasks more comparable:</b> {A['stratum_n']} target-genre / initial full-history count strata, with at least 10 profiles per group in each. Retained: 685 short and 490 long histories; {A['excluded_n']} profiles were outside this comparison.<br/><br/><b>Interpret cautiously:</b> no full-history successes occurred among the 490 retained long histories. Sparse outcomes make naive bootstrap intervals unreliable; other differences remain unadjusted.",8.5,12,maxh=75)
c.showPage()

# 5. End at an implementable design implication, with concise methods and limits.
page_start(5,'What should change?')
title('Give users control.','Show the consequences.')
para(L,174,RW,'The design implication is a reversible profile control with a visible audit. Responsiveness, relevance, genre diversity and group differences should be evaluated together.',10.7,15.5,maxh=49)
rect(L,237,RW,106,INK,10)
text(L+16,260,'PROPOSED TOOL / A PROFILE CONTROL AND AUDIT PANEL',8.5,'Bold',MINT)
para(L+16,274,RW-32,'Let a user try full history or a recent-history reset, preview the resulting choices, and undo the change. Explain the retained history and show which outcomes improve or worsen.',10.5,15.5,WHITE,maxh=51)
text(L,373,'DATA AND METHOD',9,'Bold',TEAL)
left=L;right=311;col=242
y=389
for item in [
    f"<b>Source.</b> GroupLens MovieLens 32M: {s['raw_ratings']:,} ratings from {s['raw_users']:,} users. Only ratings, timestamps and movie genres were used. Source archive and CSV checksums were verified.",
    f"<b>Training.</b> Before 1 Jan 2023, retain movies with 100+ positive ratings (4+ stars). This gives {s['catalog_movies']:,} movies and {s['training_positive_ratings']:,} positive interactions from {s['training_users']:,} users.",
    '<b>Model.</b> 32-factor SVD on binary likes, with columns scaled by inverse square-root popularity. Normalize movie vectors; sum liked-movie vectors for each profile. Rank by dot product, excluding all rated movies. Movie vectors remain fixed.',
    '<b>Steering test.</b> Select 2,000 profiles with 20+ prior likes and an eligible target: at most 10% of prior likes, 100+ catalogue movies, and fewer than 3 initial top-10 matches under both methods. Sample ten unrated target-genre movies uniformly; both methods receive the same sequence. Exclude injected movies from recommendations.'
]:y=para(left,y,col,item,8.15,11.4,maxh=110)+9
z=389
for item in [
    '<b>Exposure proxy.</b> Generated top-10 recommendations, not observed views. Fractionally divide each movie across its genre labels; exclude IMAX and missing-label tokens. Effective genres = exp(-sum p log p), using known genres only.',
    f"<b>Later-rating test.</b> A separate random sample of {B['n']:,} of {B['eligible_n']:,} users with 20+ earlier and 5+ later positive ratings. Jan-Oct 2023 positives are test labels; none train the model. nDCG@10 rewards retrieving later likes near the top.",
    '<b>Uncertainty.</b> 2,000 user-bootstrap resamples for steering and later-rating comparisons; paired differences use the same users. The model and one synthetic signal sequence per profile are fixed. Subgroup rates on page 4 are descriptive.',
    '<b>Limits and next validation.</b> Selected volunteers, a popularity-filtered catalogue and selectively observed ratings limit generalization. Equal-weight sums dilute new signals in longer histories by construction. Resetting changes content as well as length. Compare random-20 histories, recency decay and repeated signal sequences before attributing the gain to recency.'
]:z=para(right,z,col,item,8.15,11.4,maxh=110)+9
assert max(y,z)<711,(y,z)
para(L,708,RW,'No demographics, actual awareness changes or commercial-platform effects were measured. This is a test of one transparent recommender, not evidence that all feeds behave this way.',8.1,11.5,MUTED,maxh=28)
line(L,745,W-L,745)
repo=f' | <link href="{args.repo_url}" color="{TEAL}">Code and verification</link>' if args.repo_url else ''
para(L,755,RW,f'<link href="https://grouplens.org/datasets/movielens/32m/" color="{TEAL}">GroupLens / MovieLens 32M</link> | Harper &amp; Konstan (2015), <link href="https://doi.org/10.1145/2827872" color="{TEAL}">doi:10.1145/2827872</link>.<br/>Related reachability research: <link href="https://proceedings.mlr.press/v139/curmei21a.html" color="{TEAL}">Curmei, Dean &amp; Recht (ICML 2021)</link>{repo}. Code, outputs and checks accompany this report.<br/>Seeds: 20261018 (model/steering), 20261019 (extension). AI-assisted analysis and report preparation.',7.2,10.1,MUTED,maxh=34)
c.showPage();c.save()
print(args.output)
